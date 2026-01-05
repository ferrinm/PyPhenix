import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import re
from dataclasses import dataclass
import json
import warnings


@dataclass
class PhenixMetadata:
    """Container for Opera Phenix metadata"""
    plate_id: str
    plate_rows: int
    plate_columns: int
    wells: List[str]
    channels: Dict[int, Dict[str, str]]
    image_size: Tuple[int, int]
    pixel_size: Tuple[float, float]  # in meters
    z_step: Optional[float]
    timepoints: List[int]
    timepoint_offsets: np.ndarray  # time in seconds for each timepoint
    fields: List[int]
    planes: List[int]
    channel_ids: List[int]


class OperaPhenixReader:
    """
    Reader for Opera Phenix exported experiment data.
    
    Reads TIFF images and XML metadata from exported experiments,
    returning numpy arrays and metadata dictionaries.
    """
    
    def __init__(self, experiment_path: str):
        """
        Initialize reader with path to experiment directory.
        
        Parameters
        ----------
        experiment_path : str
            Path to the exported experiment directory
        """
        self.experiment_path = Path(experiment_path)
        self.images_path = self.experiment_path / "Images"
        self.index_xml_path = self.images_path / "Index.xml"
        
        if not self.images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_path}")
        if not self.index_xml_path.exists():
            raise FileNotFoundError(f"Index.xml not found: {self.index_xml_path}")
        
        # Parse metadata
        self.tree = ET.parse(self.index_xml_path)
        self.root = self.tree.getroot()
        self.ns = {'ns': '43B2A954-E3C3-47E1-B392-6635266B0DD3/HarmonyV7'}
        
        self.metadata = self._parse_metadata()
        self.image_index = self._build_image_index()
        self.well_field_map = self._build_well_field_map()

    def _parse_timepoint_offsets(self) -> np.ndarray:
        """
        Extract timepoint offsets in seconds from MeasurementTimeOffset field.
        
        Returns
        -------
        np.ndarray
            Array of time offsets in seconds, indexed by timepoint position.
            For example, array[0] is the time for TimepointID=1.
        """
        timepoint_dict = {}
        
        # Parse incrementally for memory efficiency with large XML files
        for event, elem in ET.iterparse(str(self.index_xml_path), events=('end',)):
            if elem.tag.endswith('Image'):
                # Find TimepointID and MeasurementTimeOffset
                tp_elem = elem.find('.//{*}TimepointID')
                time_elem = elem.find('.//{*}MeasurementTimeOffset')
                
                if tp_elem is not None and time_elem is not None:
                    timepoint_id = int(tp_elem.text)
                    time_offset = float(time_elem.text)
                    
                    # Only store first occurrence of each timepoint
                    if timepoint_id not in timepoint_dict:
                        timepoint_dict[timepoint_id] = time_offset
                
                # Clear element to free memory
                elem.clear()
        
        # Convert to sorted array
        if not timepoint_dict:
            return np.array([])
        
        sorted_timepoints = sorted(timepoint_dict.items())
        return np.array([time for _, time in sorted_timepoints])
    
    def _parse_metadata(self) -> PhenixMetadata:
        """Parse metadata from Index.xml"""
        # Parse plate information
        plate = self.root.find('.//ns:Plate', self.ns)
        plate_id = plate.find('ns:PlateID', self.ns).text
        plate_rows = int(plate.find('ns:PlateRows', self.ns).text)
        plate_columns = int(plate.find('ns:PlateColumns', self.ns).text)
        
        # Parse wells
        wells = [well.attrib['id'] for well in plate.findall('ns:Well', self.ns)]
        
        # Parse channel information
        channels = {}
        image_size_x = None
        image_size_y = None
        pixel_size_x = None
        pixel_size_y = None
        
        channel_maps = self.root.findall('.//ns:Maps/ns:Map', self.ns)
        for map_elem in channel_maps:
            entries = map_elem.findall('ns:Entry', self.ns)
            for entry in entries:
                # Check if this entry has channel metadata
                ch_name_elem = entry.find('ns:ChannelName', self.ns)
                if ch_name_elem is not None:
                    ch_id = int(entry.attrib['ChannelID'])
                    
                    # Extract channel information
                    channels[ch_id] = {
                        'name': ch_name_elem.text,
                        'excitation': entry.find('ns:MainExcitationWavelength', self.ns).text,
                        'emission': entry.find('ns:MainEmissionWavelength', self.ns).text,
                        'exposure': entry.find('ns:ExposureTime', self.ns).text,
                        'objective_mag': entry.find('ns:ObjectiveMagnification', self.ns).text,
                        'objective_na': entry.find('ns:ObjectiveNA', self.ns).text,
                    }
                    
                    # Get image dimensions and pixel size from first channel with this info
                    if image_size_x is None:
                        img_size_x_elem = entry.find('ns:ImageSizeX', self.ns)
                        img_size_y_elem = entry.find('ns:ImageSizeY', self.ns)
                        pix_size_x_elem = entry.find('ns:ImageResolutionX', self.ns)
                        pix_size_y_elem = entry.find('ns:ImageResolutionY', self.ns)
                        
                        if all([img_size_x_elem is not None, 
                               img_size_y_elem is not None,
                               pix_size_x_elem is not None,
                               pix_size_y_elem is not None]):
                            image_size_x = int(img_size_x_elem.text)
                            image_size_y = int(img_size_y_elem.text)
                            pixel_size_x = float(pix_size_x_elem.text)
                            pixel_size_y = float(pix_size_y_elem.text)
        
        # Fallback: if we didn't find image dimensions, check if we can get from first image
        if image_size_x is None:
            first_image = self.root.find('.//ns:Images/ns:Image', self.ns)
            if first_image is not None:
                # Will load one image to get dimensions
                url = first_image.find('ns:URL', self.ns).text
                img_path = self.images_path / url
                if img_path.exists():
                    from PIL import Image as PILImage
                    with PILImage.open(img_path) as img:
                        image_size_x = img.width
                        image_size_y = img.height
                    # Still need pixel size - use default if not found
                    pixel_size_x = pixel_size_x or 2.96688132474701E-07
                    pixel_size_y = pixel_size_y or 2.96688132474701E-07
        
        # Parse all images to determine dimensions
        images = self.root.findall('.//ns:Images/ns:Image', self.ns)
        timepoints = set()
        fields = set()
        planes = set()
        channel_ids = set()
        z_positions = set()
        
        for img in images:
            timepoints.add(int(img.find('ns:TimepointID', self.ns).text))
            fields.add(int(img.find('ns:FieldID', self.ns).text))
            planes.add(int(img.find('ns:PlaneID', self.ns).text))
            channel_ids.add(int(img.find('ns:ChannelID', self.ns).text))
            z_pos = float(img.find('ns:PositionZ', self.ns).text)
            z_positions.add(z_pos)
        
        # Calculate Z step
        z_step = None
        if len(z_positions) > 1:
            z_sorted = sorted(z_positions)
            z_step = abs(z_sorted[1] - z_sorted[0])
        
        # Extract timepoint offsets
        timepoint_offsets = self._parse_timepoint_offsets()
        
        return PhenixMetadata(
            plate_id=plate_id,
            plate_rows=plate_rows,
            plate_columns=plate_columns,
            wells=wells,
            channels=channels,
            image_size=(image_size_y, image_size_x),
            pixel_size=(pixel_size_y, pixel_size_x),
            z_step=z_step,
            timepoints=sorted(timepoints),
            timepoint_offsets=timepoint_offsets,
            fields=sorted(fields),
            planes=sorted(planes),
            channel_ids=sorted(channel_ids)
        )
    
    def _build_image_index(self) -> Dict:
        """Build index of all images for fast lookup"""
        index = {}
        images = self.root.findall('.//ns:Images/ns:Image', self.ns)
        
        for img in images:
            row = int(img.find('ns:Row', self.ns).text)
            col = int(img.find('ns:Col', self.ns).text)
            field = int(img.find('ns:FieldID', self.ns).text)
            plane = int(img.find('ns:PlaneID', self.ns).text)
            timepoint = int(img.find('ns:TimepointID', self.ns).text)
            channel = int(img.find('ns:ChannelID', self.ns).text)
            url = img.find('ns:URL', self.ns).text
            
            pos_x = float(img.find('ns:PositionX', self.ns).text)
            pos_y = float(img.find('ns:PositionY', self.ns).text)
            pos_z = float(img.find('ns:PositionZ', self.ns).text)
            
            key = (row, col, field, plane, timepoint, channel)
            index[key] = {
                'url': url,
                'position': (pos_x, pos_y, pos_z)
            }
        
        return index
    
    def _build_well_field_map(self) -> Dict:
        """Build a map of which fields exist for each well"""
        well_field_map = {}
        for key in self.image_index.keys():
            row, col, field, plane, timepoint, channel = key
            well_id = (row, col)
            if well_id not in well_field_map:
                well_field_map[well_id] = set()
            well_field_map[well_id].add(field)
        
        # Convert sets to sorted lists
        for well_id in well_field_map:
            well_field_map[well_id] = sorted(well_field_map[well_id])
        
        return well_field_map

    def format_timepoint_label(self, timepoint_idx: int) -> str:
        """
        Format a timepoint label for display.
        
        Parameters
        ----------
        timepoint_idx : int
            Zero-based index of the timepoint (0 corresponds to first timepoint)
            
        Returns
        -------
        str
            Formatted label, e.g., "T=27h 08m 15s"
        """
        if timepoint_idx >= len(self.metadata.timepoint_offsets):
            return "T=???"
        
        seconds = self.metadata.timepoint_offsets[timepoint_idx]
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        return f"T={hours:02d}h {minutes:02d}m {secs:02d}s"

    def _calculate_dataset_size(self) -> Tuple[int, str]:
        """
        Calculate expected total size of all TIFF files in the dataset based on metadata.

        Returns
        -------
        size_bytes : int
            Expected total size in bytes
        size_human : str
            Human-readable size string with breakdown
        """
        # Get image dimensions and bit depth
        img_h, img_w = self.metadata.image_size
        bytes_per_pixel = 2  # uint16

        # Calculate size per image
        pixels_per_image = img_h * img_w
        bytes_per_image = pixels_per_image * bytes_per_pixel

        # Count total number of images
        total_images = len(self.image_index)

        # Get counts for breakdown
        n_wells = len(self.metadata.wells)
        n_fields = len(self.metadata.fields)
        n_timepoints = len(self.metadata.timepoints)
        n_channels = len(self.metadata.channel_ids)
        n_z_planes = len(self.metadata.planes)

        # Calculate total expected size
        total_size = total_images * bytes_per_image

        # TIFF overhead (headers, metadata)
        tiff_overhead_factor = 1.03
        total_size_with_overhead = int(total_size * tiff_overhead_factor)

        # Convert to human-readable format
        size_value = float(total_size_with_overhead)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_value < 1024.0:
                size_human = f"{size_value:.2f} {unit}"
                break
            size_value /= 1024.0
        else:
            size_human = f"{size_value:.2f} PB"

        # Create detailed breakdown
        size_per_image_kb = bytes_per_image / 1024
        breakdown = (f"{size_human} "
                    f"({total_images:,} images × {size_per_image_kb:.1f} KB/image)")

        return total_size_with_overhead, breakdown
    
    def _print_dataset_overview(self):
        """Print overview of entire dataset"""
        print("\n" + "="*60)
        print("DATASET OVERVIEW")
        print("="*60)

        print(f"\nPlate ID: {self.metadata.plate_id}")
        print(f"Plate dimensions: {self.metadata.plate_rows} rows × {self.metadata.plate_columns} columns")

        # Calculate and display dataset size
        total_size, size_breakdown = self._calculate_dataset_size()
        print(f"Total dataset size (estimated): {size_breakdown}")

        # Optional: Show the calculation
        img_h, img_w = self.metadata.image_size
        bytes_per_pixel = 2
        total_images = len(self.image_index)
        print(f"  Calculation: {total_images:,} images × {img_h}×{img_w} pixels × {bytes_per_pixel} bytes/pixel")

        print(f"\nWells with data: {len(self.metadata.wells)}")
        print(f"  Wells: {', '.join(self.metadata.wells)}")

        print(f"\nFields per well:")
        for well_id in self.metadata.wells:
            row, col = int(well_id[:2]), int(well_id[2:])
            fields = self.well_field_map.get((row, col), [])
            print(f"  r{row:02d}c{col:02d}: {len(fields)} fields ({min(fields) if fields else 'N/A'}-{max(fields) if fields else 'N/A'})")

        print(f"\nTimepoints: {len(self.metadata.timepoints)} ({min(self.metadata.timepoints)}-{max(self.metadata.timepoints)})")
        
        # Add time information
        if len(self.metadata.timepoint_offsets) > 0:
            time_increments = np.diff(self.metadata.timepoint_offsets)
            mean_interval = time_increments.mean() if len(time_increments) > 0 else 0.0
            total_duration = self.metadata.timepoint_offsets[-1] - self.metadata.timepoint_offsets[0] if len(self.metadata.timepoint_offsets) > 1 else 0.0
            
            print(f"  First timepoint: {self.format_timepoint_label(0)}")
            print(f"  Last timepoint: {self.format_timepoint_label(len(self.metadata.timepoint_offsets) - 1)}")
            print(f"  Mean interval: {mean_interval:.1f} s ({mean_interval/60:.1f} min)")
            print(f"  Total duration: {total_duration/3600:.2f} hours")

        print(f"\nChannels: {len(self.metadata.channel_ids)}")
        for ch_id in self.metadata.channel_ids:
            ch_info = self.metadata.channels[ch_id]
            print(f"  Channel {ch_id}: {ch_info['name']}")

        print(f"\nZ-planes: {len(self.metadata.planes)} ({min(self.metadata.planes)}-{max(self.metadata.planes)})")

        print(f"\nImage dimensions: {self.metadata.image_size[0]} × {self.metadata.image_size[1]} pixels")
        print(f"Pixel size: {self.metadata.pixel_size[0]*1e6:.3f} × {self.metadata.pixel_size[1]*1e6:.3f} µm")

        if self.metadata.z_step is not None:
            print(f"Z-step: {self.metadata.z_step*1e6:.3f} µm")

        print("="*60 + "\n")
    
    def read_data(self,
                row: Optional[int] = None,
                column: Optional[int] = None,
                field: Optional[int] = None,
                stitch_fields: bool = False,
                timepoints: Optional[Union[int, List[int]]] = None,
                channels: Optional[Union[int, List[int]]] = None,
                z_slices: Optional[Union[int, List[int]]] = None,
                metadata_only: bool = False,
                output_file: Optional[str] = None,
                output_format: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
        """
        Read image data from Opera Phenix experiment.

        Parameters
        ----------
        row : int, optional
            Well row (default: first available)
        column : int, optional
            Well column (default: first available)
        field : int, optional
            Field to read (default: first available, ignored if stitching)
        stitch_fields : bool, default False
            Whether to stitch multiple fields together
        timepoints : int or list of int, optional
            Timepoint(s) to read (default: all)
        channels : int or list of int, optional
            Channel(s) to read (default: all)
        z_slices : int or list of int, optional
            Z plane(s) to read (default: all)
        metadata_only : bool, default False
            If True, only print metadata without loading image data
        output_file : str, optional
            Path to save output file
        output_format : str, optional
            Format for output file: 'ome-tiff', 'numpy', or 'parquet'

        Returns
        -------
        data : np.ndarray or None
            Image data array with dimensions (T, C, Z, Y, X), or None if metadata_only=True
        metadata : dict
            Dictionary containing metadata
        """
        # Print dataset overview first
        self._print_dataset_overview()

        if metadata_only:
            # Set defaults for metadata preparation
            if row is None:
                row = min([int(w[:2]) for w in self.metadata.wells])
            if column is None:
                column = min([int(w[2:]) for w in self.metadata.wells])

            available_fields = self.well_field_map.get((row, column), self.metadata.fields)

            if stitch_fields:
                fields = available_fields
            else:
                if field is None:
                    field = available_fields[0] if available_fields else self.metadata.fields[0]
                fields = [field]

            if timepoints is None:
                timepoints = self.metadata.timepoints
            elif isinstance(timepoints, int):
                timepoints = [timepoints]

            if channels is None:
                channels = self.metadata.channel_ids
            elif isinstance(channels, int):
                channels = [channels]

            if z_slices is None:
                z_slices = self.metadata.planes
            elif isinstance(z_slices, int):
                z_slices = [z_slices]

            # Calculate what the shape would be without loading data
            n_time = len(timepoints)
            n_channels = len(channels)
            n_z = len(z_slices)
            img_h, img_w = self.metadata.image_size

            if stitch_fields:
                # Calculate stitched dimensions
                field_positions = {}
                for fld in fields:
                    key = (row, column, fld, z_slices[0], timepoints[0], channels[0])
                    if key in self.image_index:
                        pos = self.image_index[key]['position']
                        field_positions[fld] = (pos[0], pos[1])

                if field_positions:
                    pixel_size = self.metadata.pixel_size[0]
                    positions_x = [pos[0] for pos in field_positions.values()]
                    positions_y = [pos[1] for pos in field_positions.values()]
                    min_x, max_x = min(positions_x), max(positions_x)
                    min_y, max_y = min(positions_y), max(positions_y)
                    stitched_w = int((max_x - min_x) / pixel_size) + img_w
                    stitched_h = int((max_y - min_y) / pixel_size) + img_h
                    shape = (n_time, n_channels, n_z, stitched_h, stitched_w)
                else:
                    shape = (n_time, n_channels, n_z, img_h, img_w)
            else:
                shape = (n_time, n_channels, n_z, img_h, img_w)

            # Prepare and print metadata
            metadata_dict = self._prepare_metadata_dict(
                row, column, fields, timepoints, channels, z_slices, 
                stitch_fields, shape
            )
            self._print_metadata(metadata_dict)

            print("\n*** METADATA ONLY - No image data loaded ***\n")

            return None, metadata_dict
        
        # Set defaults
        if row is None:
            row = min([int(w[:2]) for w in self.metadata.wells])
        if column is None:
            column = min([int(w[2:]) for w in self.metadata.wells])
        
        # Get available fields for this well
        available_fields = self.well_field_map.get((row, column), self.metadata.fields)
        
        if stitch_fields:
            fields = available_fields
        else:
            if field is None:
                field = available_fields[0] if available_fields else self.metadata.fields[0]
            fields = [field]
        
        if timepoints is None:
            timepoints = self.metadata.timepoints
        elif isinstance(timepoints, int):
            timepoints = [timepoints]
        
        if channels is None:
            channels = self.metadata.channel_ids
        elif isinstance(channels, int):
            channels = [channels]
        
        if z_slices is None:
            z_slices = self.metadata.planes
        elif isinstance(z_slices, int):
            z_slices = [z_slices]
        
        # Read images
        if stitch_fields:
            data = self._read_and_stitch(row, column, fields, timepoints, 
                                        channels, z_slices)
        else:
            data = self._read_images(row, column, fields, timepoints,
                                   channels, z_slices)
        
        # Prepare metadata dictionary
        metadata_dict = self._prepare_metadata_dict(
            row, column, fields, timepoints, channels, z_slices, 
            stitch_fields, data.shape
        )
        
        # Print metadata
        self._print_metadata(metadata_dict)
        
        # Save output if requested
        if output_file is not None and output_format is not None:
            self._save_output(data, metadata_dict, output_file, output_format)
        
        return data, metadata_dict
    
    def _read_images(self, row: int, col: int, fields: List[int],
                    timepoints: List[int], channels: List[int],
                    z_slices: List[int]) -> np.ndarray:
        """Read images without stitching"""
        # Determine output shape
        n_time = len(timepoints)
        n_channels = len(channels)
        n_z = len(z_slices)
        img_h, img_w = self.metadata.image_size
        
        # Initialize array (always 5D: T, C, Z, Y, X)
        data = np.zeros((n_time, n_channels, n_z, img_h, img_w),
                       dtype=np.uint16)
        
        # Track missing images
        missing_images = []
        
        # Read images (only from first field since field parameter accepts single value)
        field = fields[0]
        for t_idx, timepoint in enumerate(timepoints):
            for c_idx, channel in enumerate(channels):
                for z_idx, z_slice in enumerate(z_slices):
                    key = (row, col, field, z_slice, timepoint, channel)
                    if key in self.image_index:
                        img_path = self.images_path / self.image_index[key]['url']
                        if img_path.exists():
                            img = Image.open(img_path)
                            data[t_idx, c_idx, z_idx] = np.array(img)
                        else:
                            missing_images.append({
                                'key': key,
                                'path': str(img_path),
                                'reason': 'file not found'
                            })
                    else:
                        missing_images.append({
                            'key': key,
                            'path': f"r{row:02d}c{col:02d}f{field:02d}p{z_slice:02d}-ch{channel}sk1fk1fl1.tiff",
                            'reason': 'not in index'
                        })
        
        # Print warnings for missing images
        if missing_images:
            print("\n" + "!"*60)
            print(f"WARNING: {len(missing_images)} missing images")
            print("!"*60)
            for miss in missing_images[:10]:  # Show first 10
                row, col, field, plane, timepoint, channel = miss['key']
                print(f"  Missing: r{row:02d}c{col:02d}f{field:02d}p{plane:02d}t{timepoint}ch{channel}")
                print(f"    Reason: {miss['reason']}")
            if len(missing_images) > 10:
                print(f"  ... and {len(missing_images) - 10} more")
            print("  These positions will be filled with zeros.")
            print("!"*60 + "\n")
        
        return data
    
    def _read_and_stitch(self, row: int, col: int, fields: List[int],
                        timepoints: List[int], channels: List[int],
                        z_slices: List[int]) -> np.ndarray:
        """Read and stitch multiple fields"""
        # Get field positions
        field_positions = {}
        for field in fields:
            key = (row, col, field, z_slices[0], timepoints[0], channels[0])
            if key in self.image_index:
                pos = self.image_index[key]['position']
                field_positions[field] = (pos[0], pos[1])
        
        if not field_positions:
            raise ValueError(f"No valid field positions found for well r{row:02d}c{col:02d}")
        
        # Calculate stitched dimensions
        img_h, img_w = self.metadata.image_size
        pixel_size = self.metadata.pixel_size[0]  # assume square pixels
        
        positions_x = [pos[0] for pos in field_positions.values()]
        positions_y = [pos[1] for pos in field_positions.values()]
        
        min_x, max_x = min(positions_x), max(positions_x)
        min_y, max_y = min(positions_y), max(positions_y)
        
        stitched_w = int((max_x - min_x) / pixel_size) + img_w
        stitched_h = int((max_y - min_y) / pixel_size) + img_h
        
        # Initialize stitched array
        n_time = len(timepoints)
        n_channels = len(channels)
        n_z = len(z_slices)
        
        data = np.zeros((n_time, n_channels, n_z, stitched_h, stitched_w),
                       dtype=np.uint16)
        
        # Track missing images
        missing_images = []
        
        # Stitch images with maximum intensity projection for overlaps
        for t_idx, timepoint in enumerate(timepoints):
            for c_idx, channel in enumerate(channels):
                for z_idx, z_slice in enumerate(z_slices):
                    for field in fields:
                        key = (row, col, field, z_slice, timepoint, channel)
                        if key in self.image_index:
                            img_path = self.images_path / self.image_index[key]['url']
                            if img_path.exists():
                                img = np.array(Image.open(img_path))

                                # Calculate position in stitched image
                                pos = field_positions[field]
                                x_offset = int((pos[0] - min_x) / pixel_size)
                                y_offset = int((max_y - pos[1]) / pixel_size)

                                # Use maximum intensity projection for overlaps
                                current_region = data[t_idx, c_idx, z_idx,
                                                    y_offset:y_offset+img_h,
                                                    x_offset:x_offset+img_w]
                                data[t_idx, c_idx, z_idx,
                                    y_offset:y_offset+img_h,
                                    x_offset:x_offset+img_w] = np.maximum(current_region, img)
                            else:
                                missing_images.append({
                                    'key': key,
                                    'path': str(img_path),
                                    'reason': 'file not found'
                                })
                        else:
                            missing_images.append({
                                'key': key,
                                'path': f"r{row:02d}c{col:02d}f{field:02d}p{z_slice:02d}-ch{channel}sk1fk1fl1.tiff",
                                'reason': 'not in index'
                            })
        
        # Print warnings for missing images
        if missing_images:
            print("\n" + "!"*60)
            print(f"WARNING: {len(missing_images)} missing images")
            print("!"*60)
            for miss in missing_images[:10]:  # Show first 10
                row, col, field, plane, timepoint, channel = miss['key']
                print(f"  Missing: r{row:02d}c{col:02d}f{field:02d}p{plane:02d}t{timepoint}ch{channel}")
                print(f"    Reason: {miss['reason']}")
            if len(missing_images) > 10:
                print(f"  ... and {len(missing_images) - 10} more")
            print("  These positions will be filled with zeros.")
            print("!"*60 + "\n")
        
        return data
    
    def _prepare_metadata_dict(self, row: int, col: int, fields: List[int],
                               timepoints: List[int], channels: List[int],
                               z_slices: List[int], stitched: bool,
                               shape: Tuple) -> Dict:
        """Prepare metadata dictionary for output"""
        channel_info = {ch: self.metadata.channels[ch] for ch in channels}
        
        # Get time offsets for selected timepoints (convert to 0-based indexing)
        selected_time_offsets = []
        for tp in timepoints:
            tp_idx = tp - 1  # Convert TimepointID to 0-based index
            if 0 <= tp_idx < len(self.metadata.timepoint_offsets):
                selected_time_offsets.append(float(self.metadata.timepoint_offsets[tp_idx]))
        
        # Calculate time increment if multiple timepoints
        time_increment = None
        if len(selected_time_offsets) > 1:
            increments = np.diff(selected_time_offsets)
            time_increment = float(increments.mean())
        
        metadata = {
            'plate_id': self.metadata.plate_id,
            'plate_layout': {
                'rows': self.metadata.plate_rows,
                'columns': self.metadata.plate_columns
            },
            'well': f"r{row:02d}c{col:02d}",
            'shape': {
                'description': 'T, C, Z, Y, X',
                'dimensions': shape
            },
            'fields': fields,
            'timepoints': timepoints,
            'timepoint_offsets': selected_time_offsets,
            'time_increment': time_increment,
            'time_unit': 's',
            'channels': channel_info,
            'z_slices': z_slices,
            'pixel_size': {
                'x': self.metadata.pixel_size[1],
                'y': self.metadata.pixel_size[0],
                'unit': 'm'
            },
            'z_step': self.metadata.z_step,
            'stitched': stitched
        }
        
        return metadata
    
    def _print_metadata(self, metadata: dict):
        """print metadata to console"""
        print("\n" + "="*60)
        print("loaded data summary")
        print("="*60)

        print(f"\nplate id: {metadata['plate_id']}")
        print(f"well: {metadata['well']}")

        print(f"\ndata shape: {metadata['shape']['dimensions']}")
        print(f"  dimension order: {metadata['shape']['description']}")

        # calculate loaded data size in memory
        shape = metadata['shape']['dimensions']
        n_pixels = np.prod(shape)
        bytes_per_pixel = 2  # uint16
        total_bytes = n_pixels * bytes_per_pixel

        # convert to human-readable
        size_value = total_bytes
        for unit in ['b', 'kb', 'mb', 'gb', 'tb']:
            if size_value < 1024.0:
                size_human = f"{size_value:.2f} {unit}"
                break
            size_value /= 1024.0
        else:
            size_human = f"{size_value:.2f} pb"

        print(f"  loaded data size in memory: {size_human}")

        print(f"\nchannels:")
        for ch_id, ch_info in metadata['channels'].items():
            print(f"  channel {ch_id}: {ch_info['name']}")
            print(f"    excitation: {ch_info['excitation']} nm")
            print(f"    emission: {ch_info['emission']} nm")
            print(f"    exposure: {ch_info['exposure']} s")

        print(f"\nfields: {metadata['fields']}")
        print(f"timepoints: {metadata['timepoints']}")
        if metadata['timepoint_offsets']:
            print(f"  timepoint offsets (s): {[f'{t:.1f}' for t in metadata['timepoint_offsets'][:5]]}")
            if len(metadata['timepoint_offsets']) > 5:
                print(f"    ... and {len(metadata['timepoint_offsets']) - 5} more")
            if metadata['time_increment'] is not None:
                print(f"  mean time increment: {metadata['time_increment']:.1f} s ({metadata['time_increment']/60:.1f} min)")
        print(f"z-slices: {metadata['z_slices']}")

        print(f"\nphysical dimensions:")
        print(f"  pixel size (x): {metadata['pixel_size']['x']*1e6:.3f} µm")
        print(f"  pixel size (y): {metadata['pixel_size']['y']*1e6:.3f} µm")
        if metadata['z_step'] is not None:
            print(f"  z step: {metadata['z_step']*1e6:.3f} µm")

        if metadata['stitched']:
            print(f"\n*** fields have been stitched ***")

        print("="*60 + "\n")
    
    def _save_output(self, data: np.ndarray, metadata: Dict,
                    output_file: str, output_format: str):
        """Save data and metadata to file"""
        output_path = Path(output_file)
        
        if output_format == 'numpy':
            np.save(output_path, data)
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            print(f"Saved numpy array to: {output_path}")
            print(f"Saved metadata to: {metadata_path}")
        
        elif output_format == 'ome-tiff':
            try:
                import tifffile
                tifffile.imwrite(output_path, data, photometric='minisblack')
                metadata_path = output_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                print(f"Saved OME-TIFF to: {output_path}")
                print(f"Saved metadata to: {metadata_path}")
            except ImportError:
                print("Warning: tifffile not available. Falling back to numpy format.")
                self._save_output(data, metadata, str(output_path.with_suffix('.npy')), 'numpy')
        
        elif output_format == 'parquet':
            print("Warning: Parquet format not implemented for image data.")
            print("Saving as numpy instead.")
            self._save_output(data, metadata, str(output_path.with_suffix('.npy')), 'numpy')

def napari_get_reader(path):
    """
    A napari reader plugin for Opera Phenix experiments.
    
    Parameters
    ----------
    path : str or list of str
        Path to file or directory, or list of paths.
    
    Returns
    -------
    function or None
        If the path is recognized, returns a reader function.
        Otherwise returns None.
    """
    # Handle both string and Path objects
    if isinstance(path, list):
        path = path[0]
    
    path = Path(path)
    
    # Check if this looks like a Phenix experiment directory
    if path.is_dir():
        images_path = path / "Images"
        index_xml_path = images_path / "Index.xml"
        
        if images_path.exists() and index_xml_path.exists():
            return phenix_reader
    
    return None


def phenix_reader(path):
    """
    Read Opera Phenix experiment data.
    
    Parameters
    ----------
    path : str or Path
        Path to the experiment directory
    
    Returns
    -------
    layer_data : list of tuples
        List of (data, metadata, layer_type) tuples for napari
    """
    reader = OperaPhenixReader(str(path))
    
    # Load data with defaults (first well, all channels, etc.)
    data, metadata = reader.read_data()
    
    # Prepare layers for napari
    layer_data_list = []
    
    # Extract metadata
    channels_info = metadata['channels']
    pixel_size_x = metadata['pixel_size']['x'] * 1e6  # to µm
    pixel_size_y = metadata['pixel_size']['y'] * 1e6
    z_step = metadata['z_step'] * 1e6 if metadata['z_step'] is not None else 1.0
    
    # Color mapping
    color_map = {
        'DAPI': 'blue',
        'Hoechst': 'blue',
        'Alexa 488': 'green',
        'GFP': 'green',
        'Alexa 555': 'yellow',
        'mCherry': 'red',
        'Alexa 647': 'magenta',
        'Cy5': 'magenta',
    }
    default_colors = ['cyan', 'magenta', 'yellow', 'green', 'red', 'blue']
    
    # Add each channel as a layer
    for ch_idx, (ch_id, ch_info) in enumerate(channels_info.items()):
        ch_name = ch_info['name']
        
        # Select color
        color = None
        for key, value in color_map.items():
            if key.lower() in ch_name.lower():
                color = value
                break
        if color is None:
            color = default_colors[ch_idx % len(default_colors)]
        
        # Get channel data
        if data.shape[0] > 1:  # Multiple timepoints
            channel_data = data[:, ch_idx, :, :, :]
            scale = (1, z_step, pixel_size_y, pixel_size_x)
        else:
            channel_data = data[0, ch_idx, :, :, :]
            scale = (z_step, pixel_size_y, pixel_size_x)
        
        # Calculate contrast limits
        nonzero_data = channel_data[channel_data > 0]
        if len(nonzero_data) > 0:
            contrast_limits = [0, np.percentile(nonzero_data, 99.5)]
        else:
            contrast_limits = [0, 1]
        
        # Create layer metadata
        layer_meta = {
            'name': f"Ch{ch_id}: {ch_name}",
            'colormap': color,
            'blending': 'additive',
            'scale': scale,
            'contrast_limits': contrast_limits,
        }
        
        layer_data_list.append((channel_data, layer_meta, 'image'))
    
    return layer_data_list
