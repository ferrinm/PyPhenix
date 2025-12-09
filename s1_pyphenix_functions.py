import napari
import numpy as np
import os
import pandas as pd
from bioio import BioImage
import xml.etree.ElementTree as ET
import json


def load_image_stack_with_metadata(base_path, target_row, target_col, target_field):
    """
    Load image stack and extract metadata from Phenix microscope data.
    
    Parameters
    ----------
    base_path : str
        Path to the base directory containing images folder and metadata files
    target_row : int
        Target row in the plate
    target_col : int
        Target column in the plate
    target_field : int
        Target field number
        
    Returns
    -------
    image_stack : np.ndarray
        5D numpy array with shape (T, C, Z, Y, X)
    metadata : dict
        Dictionary containing extracted metadata including:
        - 'channel_names': list of channel names
        - 'scale_x': X voxel size in µm
        - 'scale_y': Y voxel size in µm
        - 'scale_z': Z voxel size in µm
        - 'num_times': number of timepoints
        - 'num_channels': number of channels
        - 'num_planes': number of Z planes
        - 'field_df': DataFrame with file information
    """
    images_path = os.path.join(base_path, 'images')
    
    # =================================================================
    # PARSE METADATA FILES
    # =================================================================
    print("Reading metadata files...")
    
    # --- A) Parse image.index.txt ---
    index_file_path = os.path.join(images_path, 'image.index.txt')
    index_df = pd.read_csv(index_file_path, sep='\t', skiprows=2)
    
    cols_to_convert = ['Row', 'Column', 'Field', 'AbsoluteZ', 'Timepoint', 'Channel', 'Plane']
    for col in cols_to_convert:
        index_df[col] = pd.to_numeric(index_df[col], errors='coerce')
    
    index_df.dropna(subset=cols_to_convert, inplace=True)
    for col in ['Row', 'Column', 'Field', 'Timepoint', 'Channel', 'Plane']:
        index_df[col] = index_df[col].astype(int)
    
    field_df = index_df[
        (index_df['Row'] == target_row) &
        (index_df['Column'] == target_col) &
        (index_df['Field'] == target_field)
    ].copy()
    
    if field_df.empty:
        raise ValueError(
            f"No data found for Row={target_row}, Column={target_col}, Field={target_field}. "
            "Please check your target values and file paths."
        )
    
    # --- B) Parse the .kw.txt file for channel names ---
    kw_files = [f for f in os.listdir(base_path) if f.endswith('.kw.txt')]
    channel_names = []
    
    if kw_files:
        kw_file_path = os.path.join(base_path, kw_files[0])
        try:
            with open(kw_file_path, 'r') as f:
                full_content = f.read()
                start, end = full_content.find('{'), full_content.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = full_content[start:end]
                    metadata_kw = json.loads(json_str)
                    channel_names = metadata_kw.get('CHANNEL', [])
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"⚠️ Warning: Could not parse .kw.txt file. {e}")
    
    # --- C) Extract Voxel Sizes from Metadata ---
    scale_x, scale_y, scale_z = 1.0, 1.0, 1.0  # Default values
    
    xml_file_path = os.path.join(images_path, 'index.xml')
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        ns = {'h': '43B2A954-E3C3-47E1-B392-6635266B0DD3/HarmonyV7'}
        
        res_x_element = root.find('.//h:ImageResolutionX', ns)
        res_y_element = root.find('.//h:ImageResolutionY', ns)

        binning_x_element = root.find('.//h:BinningX', ns)
        binning_y_element = root.find('.//h:BinningY', ns)
        
        if res_x_element is not None and res_y_element is not None:
            scale_x = float(res_x_element.text) / float(binning_x_element.text) * 1e6  # Convert from meters to µm
            scale_y = float(res_y_element.text) / float(binning_y_element.text) * 1e6  # Convert from meters to µm
        else:
            print("⚠️ Warning: Could not find ImageResolutionX/Y tags in index.xml.")
            
    except (FileNotFoundError, ET.ParseError) as e:
        print(f"⚠️ Warning: Could not parse index.xml file. {e}")
    
    # Calculate Z voxel size from the AbsoluteZ column
    num_planes_float = field_df['Plane'].nunique()
    if num_planes_float > 1:
        min_z = field_df['AbsoluteZ'].min()
        max_z = field_df['AbsoluteZ'].max()
        scale_z = ((max_z - min_z) / (num_planes_float - 1)) * 1e6  # Convert from meters to µm
    else:
        scale_z = 1.0
    
    print("\nSuccessfully parsed metadata:")
    print(f"  - Channel Names: {channel_names}")
    print(f"  - Voxel Size (Z, Y, X) in µm: ({scale_z:.4f}, {scale_y:.4f}, {scale_x:.4f})")
    
    # =================================================================
    # ASSEMBLE THE 5D NUMPY ARRAY
    # =================================================================
    num_times = int(field_df['Timepoint'].max())
    num_channels = int(field_df['Channel'].max())
    num_planes = int(field_df['Plane'].nunique())
    
    first_img_path = os.path.join(images_path, field_df.iloc[0]['__URL'])
    first_img = BioImage(first_img_path)
    height, width = first_img.dims.Y, first_img.dims.X
    
    image_stack = np.zeros(
        (num_times, num_channels, num_planes, height, width),
        dtype=first_img.dtype
    )
    
    print("\nAssembling the 5D image array...")
    plane_map = {plane: i for i, plane in enumerate(sorted(field_df['Plane'].unique()))}
    
    for _, row in field_df.iterrows():
        t = row['Timepoint'] - 1
        c = row['Channel'] - 1
        z = plane_map[row['Plane']]
        
        img_path = os.path.join(images_path, row['__URL'])
        img_data = BioImage(img_path).data
        image_stack[t, c, z, :, :] = img_data
    
    print("✅ Assembly complete.")
    
    # Prepare metadata dictionary
    metadata = {
        'channel_names': channel_names,
        'scale_x': scale_x,
        'scale_y': scale_y,
        'scale_z': scale_z,
        'num_times': num_times,
        'num_channels': num_channels,
        'num_planes': num_planes,
        'field_df': field_df,
        'target_row': target_row,
        'target_col': target_col,
        'target_field': target_field
    }
    
    return image_stack, metadata

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
    
    def _print_dataset_overview(self):
        """Print overview of entire dataset"""
        print("\n" + "="*60)
        print("DATASET OVERVIEW")
        print("="*60)
        
        print(f"\nPlate ID: {self.metadata.plate_id}")
        print(f"Plate dimensions: {self.metadata.plate_rows} rows × {self.metadata.plate_columns} columns")
        
        print(f"\nWells with data: {len(self.metadata.wells)}")
        print(f"  Wells: {', '.join(self.metadata.wells)}")
        
        print(f"\nFields per well:")
        for well_id in self.metadata.wells:
            row, col = int(well_id[:2]), int(well_id[2:])
            fields = self.well_field_map.get((row, col), [])
            print(f"  r{row:02d}c{col:02d}: {len(fields)} fields ({min(fields) if fields else 'N/A'}-{max(fields) if fields else 'N/A'})")
        
        print(f"\nTimepoints: {len(self.metadata.timepoints)} ({min(self.metadata.timepoints)}-{max(self.metadata.timepoints)})")
        
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
                metadata_only: bool = False,  # NEW PARAMETER
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
        
        # Stitch images
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
                                # FIX: Invert y-offset calculation
                                y_offset = int((max_y - pos[1]) / pixel_size)
                                
                                data[t_idx, c_idx, z_idx,
                                    y_offset:y_offset+img_h,
                                    x_offset:x_offset+img_w] = img
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
    
    def _print_metadata(self, metadata: Dict):
        """Print metadata to console"""
        print("\n" + "="*60)
        print("LOADED DATA SUMMARY")
        print("="*60)
        
        print(f"\nPlate ID: {metadata['plate_id']}")
        print(f"Well: {metadata['well']}")
        
        print(f"\nData Shape: {metadata['shape']['dimensions']}")
        print(f"  Dimension order: {metadata['shape']['description']}")
        
        print(f"\nChannels:")
        for ch_id, ch_info in metadata['channels'].items():
            print(f"  Channel {ch_id}: {ch_info['name']}")
            print(f"    Excitation: {ch_info['excitation']} nm")
            print(f"    Emission: {ch_info['emission']} nm")
            print(f"    Exposure: {ch_info['exposure']} s")
        
        print(f"\nFields: {metadata['fields']}")
        print(f"Timepoints: {metadata['timepoints']}")
        print(f"Z-slices: {metadata['z_slices']}")
        
        print(f"\nPhysical Dimensions:")
        print(f"  Pixel size (X): {metadata['pixel_size']['x']*1e6:.3f} µm")
        print(f"  Pixel size (Y): {metadata['pixel_size']['y']*1e6:.3f} µm")
        if metadata['z_step'] is not None:
            print(f"  Z step: {metadata['z_step']*1e6:.3f} µm")
        
        if metadata['stitched']:
            print(f"\n*** Fields have been STITCHED ***")
        
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


def load_phenix_data(experiment_path: str, **kwargs) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to load Opera Phenix data.
    
    Parameters
    ----------
    experiment_path : str
        Path to experiment directory
    **kwargs
        Additional arguments passed to OperaPhenixReader.read_data()
    
    Returns
    -------
    data : np.ndarray
        Image data array with dimensions (T, C, Z, Y, X)
    metadata : dict
        Metadata dictionary
    """
    reader = OperaPhenixReader(experiment_path)
    return reader.read_data(**kwargs)

import napari
import numpy as np


def visualize_in_napari(data, metadata, time_index=0):
    """
    Visualize Opera Phenix image data in Napari with proper channel names, colors, and scaling.
    
    Parameters
    ----------
    data : np.ndarray
        5D numpy array with shape (T, C, Z, Y, X) from load_phenix_data
    metadata : dict
        Dictionary containing metadata from load_phenix_data
    time_index : int, optional
        Which timepoint to display (default: 0)
        
    Returns
    -------
    viewer : napari.Viewer
        Napari viewer instance
    """
    if data is None:
        raise ValueError("Cannot visualize: data is None (metadata_only mode?)")
    
    # Extract metadata
    well = metadata['well']
    channels_info = metadata['channels']
    pixel_size_x = metadata['pixel_size']['x']  # in meters
    pixel_size_y = metadata['pixel_size']['y']  # in meters
    z_step = metadata['z_step']  # in meters, can be None
    stitched = metadata['stitched']
    fields = metadata['fields']
    
    # Convert physical scales to micrometers for Napari
    scale_x = pixel_size_x * 1e6  # meters to micrometers
    scale_y = pixel_size_y * 1e6
    scale_z = z_step * 1e6 if z_step is not None else 1.0
    
    # Validate time_index
    if time_index >= data.shape[0]:
        print(f"Warning: time_index {time_index} exceeds available timepoints ({data.shape[0]})")
        print(f"Using time_index=0 instead")
        time_index = 0
    
    # Select timepoint to view
    data_to_view = data[time_index]  # Shape: (C, Z, Y, X)
    
    # Create viewer title
    if stitched:
        title = f"{metadata['plate_id']} - {well} - Stitched ({len(fields)} fields)"
    else:
        title = f"{metadata['plate_id']} - {well} - Field {fields[0]}"
    
    viewer = napari.Viewer(title=title)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "µm"
    
    # Define colors for common fluorophores
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
    
    # Add each channel to the viewer
    for ch_idx, (ch_id, ch_info) in enumerate(channels_info.items()):
        ch_name = ch_info['name']
        
        # Select color based on channel name
        color = None
        for key, value in color_map.items():
            if key.lower() in ch_name.lower():
                color = value
                break
        if color is None:
            color = default_colors[ch_idx % len(default_colors)]
        
        # Get channel data
        channel_data = data_to_view[ch_idx]  # Shape: (Z, Y, X)
        
        # Add to viewer
        viewer.add_image(
            channel_data,
            name=f"Ch{ch_id}: {ch_name}",
            colormap=color,
            blending='additive',
            scale=(scale_z, scale_y, scale_x),
            contrast_limits=[0, np.percentile(channel_data[channel_data > 0], 99.5)] if channel_data.max() > 0 else [0, 1]
        )
    
    # Print summary
    print("\n" + "="*60)
    print("🚀 NAPARI VIEWER LAUNCHED")
    print("="*60)
    print(f"Viewing: {title}")
    print(f"Timepoint: {time_index + 1} / {data.shape[0]}")
    print(f"Channels: {len(channels_info)}")
    print(f"Z-slices: {data.shape[2]}")
    print(f"Image size: {data.shape[3]} × {data.shape[4]} pixels")
    print(f"Pixel size: {scale_x:.3f} × {scale_y:.3f} µm")
    if z_step is not None:
        print(f"Z-step: {scale_z:.3f} µm")
    print("="*60 + "\n")
    
    return viewer

import napari
from napari.utils import notifications
import numpy as np
from pathlib import Path
from magicgui import magicgui
from typing import List
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QComboBox, QListWidget, QLabel, QCheckBox,
                            QGroupBox, QAbstractItemView, QLineEdit)


class PhenixDataLoaderWidget(QWidget):
    """
    Interactive widget for loading and visualizing Opera Phenix data in Napari.
    """
    
    def __init__(self, experiment_path: str, viewer: napari.Viewer = None):
        """
        Initialize the data loader widget.
        
        Parameters
        ----------
        experiment_path : str
            Path to the Opera Phenix experiment directory
        viewer : napari.Viewer, optional
            Napari viewer instance. If None, creates new viewer.
        """
        super().__init__()
        
        self.experiment_path = experiment_path
        self.viewer = viewer if viewer is not None else napari.Viewer()
        
        # Initialize reader
        self.reader = OperaPhenixReader(experiment_path)
        self.metadata = self.reader.metadata
        
        # Build the widget UI
        self._build_ui()
        
        # Set default selections
        self._set_defaults()
        
        print("\n" + "="*60)
        print("🎛️  INTERACTIVE DATA LOADER WIDGET INITIALIZED")
        print("="*60)
        print(f"Experiment: {Path(experiment_path).name}")
        print(f"Wells available: {len(self.metadata.wells)}")
        print(f"Channels: {len(self.metadata.channels)}")
        print("="*60 + "\n")
    
    def _build_ui(self):
        """Build the user interface."""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel(f"<h2>Opera Phenix Data Loader</h2>")
        layout.addWidget(title)
        
        # Experiment info
        exp_name = Path(self.experiment_path).name
        exp_label = QLabel(f"<b>Experiment:</b> {exp_name}")
        layout.addWidget(exp_label)
        
        # Well selector
        well_group = QGroupBox("Well Selection")
        well_layout = QVBoxLayout()
        
        self.well_combo = QComboBox()
        self.well_combo.addItems([f"r{w[:2]}c{w[2:]}" for w in self.metadata.wells])
        self.well_combo.currentTextChanged.connect(self._on_well_changed)
        well_layout.addWidget(QLabel("Select Well:"))
        well_layout.addWidget(self.well_combo)
        
        well_group.setLayout(well_layout)
        layout.addWidget(well_group)
        
        # Field selector
        field_group = QGroupBox("Field Selection")
        field_layout = QVBoxLayout()
        
        self.stitch_checkbox = QCheckBox("Stitch all fields")
        self.stitch_checkbox.stateChanged.connect(self._on_stitch_changed)
        field_layout.addWidget(self.stitch_checkbox)
        
        field_layout.addWidget(QLabel("Select Field:"))
        self.field_combo = QComboBox()
        field_layout.addWidget(self.field_combo)
        
        field_group.setLayout(field_layout)
        layout.addWidget(field_group)
        
        # Timepoint selector
        time_group = QGroupBox("Timepoint Selection")
        time_layout = QVBoxLayout()
        
        time_buttons = QHBoxLayout()
        self.time_select_all_btn = QPushButton("Select All")
        self.time_select_all_btn.clicked.connect(lambda: self._select_all(self.time_list))
        self.time_clear_all_btn = QPushButton("Clear All")
        self.time_clear_all_btn.clicked.connect(lambda: self._clear_all(self.time_list))
        time_buttons.addWidget(self.time_select_all_btn)
        time_buttons.addWidget(self.time_clear_all_btn)
        time_layout.addLayout(time_buttons)
        
        self.time_list = QListWidget()
        self.time_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.time_list.addItems([f"Timepoint {t}" for t in self.metadata.timepoints])
        time_layout.addWidget(self.time_list)
        
        time_group.setLayout(time_layout)
        layout.addWidget(time_group)
        
        # Channel selector
        channel_group = QGroupBox("Channel Selection")
        channel_layout = QVBoxLayout()
        
        channel_buttons = QHBoxLayout()
        self.channel_select_all_btn = QPushButton("Select All")
        self.channel_select_all_btn.clicked.connect(lambda: self._select_all(self.channel_list))
        self.channel_clear_all_btn = QPushButton("Clear All")
        self.channel_clear_all_btn.clicked.connect(lambda: self._clear_all(self.channel_list))
        channel_buttons.addWidget(self.channel_select_all_btn)
        channel_buttons.addWidget(self.channel_clear_all_btn)
        channel_layout.addLayout(channel_buttons)
        
        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for ch_id in self.metadata.channel_ids:
            ch_name = self.metadata.channels[ch_id]['name']
            self.channel_list.addItem(f"Ch{ch_id}: {ch_name}")
        channel_layout.addWidget(self.channel_list)
        
        channel_group.setLayout(channel_layout)
        layout.addWidget(channel_group)
        
        # Z-slice selector
        z_group = QGroupBox("Z-slice Selection")
        z_layout = QVBoxLayout()
        
        z_buttons = QHBoxLayout()
        self.z_select_all_btn = QPushButton("Select All")
        self.z_select_all_btn.clicked.connect(lambda: self._select_all(self.z_list))
        self.z_clear_all_btn = QPushButton("Clear All")
        self.z_clear_all_btn.clicked.connect(lambda: self._clear_all(self.z_list))
        z_buttons.addWidget(self.z_select_all_btn)
        z_buttons.addWidget(self.z_clear_all_btn)
        z_layout.addLayout(z_buttons)
        
        self.z_list = QListWidget()
        self.z_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.z_list.addItems([f"Z-plane {z}" for z in self.metadata.planes])
        z_layout.addWidget(self.z_list)
        
        z_group.setLayout(z_layout)
        layout.addWidget(z_group)
        
        # Add save options before visualize button
        save_group = QGroupBox("Save Options")
        save_layout = QVBoxLayout()
        
        self.save_checkbox = QCheckBox("Save loaded data")
        save_layout.addWidget(self.save_checkbox)
        
        save_path_layout = QHBoxLayout()
        self.save_path_input = QLineEdit()
        self.save_path_input.setPlaceholderText("Output file path...")
        self.save_browse_btn = QPushButton("Browse")
        self.save_browse_btn.clicked.connect(self._browse_save_path)
        save_path_layout.addWidget(self.save_path_input)
        save_path_layout.addWidget(self.save_browse_btn)
        save_layout.addLayout(save_path_layout)
        
        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems(["numpy", "ome-tiff"])
        save_layout.addWidget(QLabel("Save format:"))
        save_layout.addWidget(self.save_format_combo)
        
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)
        
        # Visualize button
        self.visualize_btn = QPushButton("Visualize Data")
        self.visualize_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.visualize_btn.clicked.connect(self._visualize_data)
        layout.addWidget(self.visualize_btn)
        
        self.setLayout(layout)
    
    def _browse_save_path(self):
        """Open file dialog for save path."""
        from qtpy.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Data",
            "",
            "Numpy files (*.npy);;TIFF files (*.tiff *.tif)"
        )
        
        if file_path:
            self.save_path_input.setText(file_path)
            
    def _set_defaults(self):
        """Set default selections."""
        # Select first well
        self.well_combo.setCurrentIndex(0)
        
        # Update field selector
        self._update_field_selector()
        
        # Select first timepoint
        self.time_list.item(0).setSelected(True)
        
        # Select all channels
        self._select_all(self.channel_list)
        
        # Select all Z-slices
        self._select_all(self.z_list)
    
    def _on_well_changed(self):
        """Handle well selection change."""
        self._update_field_selector()
    
    def _on_stitch_changed(self):
        """Handle stitch checkbox change."""
        self.field_combo.setEnabled(not self.stitch_checkbox.isChecked())
    
    def _update_field_selector(self):
        """Update field selector based on selected well."""
        well_str = self.well_combo.currentText()
        row = int(well_str[1:3])
        col = int(well_str[4:6])
        
        available_fields = self.reader.well_field_map.get((row, col), self.metadata.fields)
        
        self.field_combo.clear()
        self.field_combo.addItems([f"Field {f}" for f in available_fields])
    
    def _select_all(self, list_widget: QListWidget):
        """Select all items in a list widget."""
        for i in range(list_widget.count()):
            list_widget.item(i).setSelected(True)
    
    def _clear_all(self, list_widget: QListWidget):
        """Clear all selections in a list widget."""
        list_widget.clearSelection()
    
    def _get_selected_indices(self, list_widget: QListWidget) -> List[int]:
        """Get list of selected indices from a list widget."""
        return [i.row() for i in list_widget.selectedIndexes()]
    
    def _visualize_data(self):
        """Load, visualize, and optionally save selected data."""
        # Get selections
        well_str = self.well_combo.currentText()
        row = int(well_str[1:3])
        col = int(well_str[4:6])
        
        stitch = self.stitch_checkbox.isChecked()
        
        if not stitch:
            field_str = self.field_combo.currentText()
            field = int(field_str.split()[1])
        else:
            field = None
        
        time_indices = self._get_selected_indices(self.time_list)
        if not time_indices:
            notifications.show_warning("No timepoints selected")
            return
        timepoints = [self.metadata.timepoints[i] for i in time_indices]
        
        channel_indices = self._get_selected_indices(self.channel_list)
        if not channel_indices:
            notifications.show_warning("No channels selected")
            return
        channels = [self.metadata.channel_ids[i] for i in channel_indices]
        
        z_indices = self._get_selected_indices(self.z_list)
        if not z_indices:
            notifications.show_warning("No Z-slices selected")
            return
        z_slices = [self.metadata.planes[i] for i in z_indices]
        
        # Show loading message
        notifications.show_info(f"Loading data for well {well_str}...")
        
        try:
            # Determine save parameters
            save_file = None
            save_format = None
            
            if self.save_checkbox.isChecked():
                save_file = self.save_path_input.text()
                if save_file:
                    save_format = self.save_format_combo.currentText()
                else:
                    notifications.show_warning("Save enabled but no path specified")
            
            # Load data
            data, metadata = self.reader.read_data(
                row=row,
                column=col,
                field=field,
                stitch_fields=stitch,
                timepoints=timepoints,
                channels=channels,
                z_slices=z_slices,
                output_file=save_file,
                output_format=save_format
            )
            
            # Clear existing layers
            self.viewer.layers.clear()
            
            # Visualize data
            self._add_layers_to_viewer(data, metadata)
            
            notifications.show_info("Data loaded successfully!")
            
        except Exception as e:
            notifications.show_error(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _add_layers_to_viewer(self, data, metadata):
        """Add data layers to the viewer."""
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
        
        # Add each channel
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
                channel_data = data[:, ch_idx, :, :, :]  # (T, Z, Y, X)
                scale = (1, z_step, pixel_size_y, pixel_size_x)
            else:  # Single timepoint
                channel_data = data[0, ch_idx, :, :, :]  # (Z, Y, X)
                scale = (z_step, pixel_size_y, pixel_size_x)
            
            # Calculate contrast limits
            nonzero_data = channel_data[channel_data > 0]
            if len(nonzero_data) > 0:
                contrast_limits = [0, np.percentile(nonzero_data, 99.5)]
            else:
                contrast_limits = [0, 1]
            
            # Add to viewer
            self.viewer.add_image(
                channel_data,
                name=f"Ch{ch_id}: {ch_name}",
                colormap=color,
                blending='additive',
                scale=scale,
                contrast_limits=contrast_limits
            )
        
        # Update viewer title
        well = metadata['well']
        if metadata['stitched']:
            title = f"{metadata['plate_id']} - {well} - Stitched"
        else:
            title = f"{metadata['plate_id']} - {well} - Field {metadata['fields'][0]}"
        self.viewer.title = title
        
        # Enable scale bar
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = "µm"
        
        # Reset view
        self.viewer.reset_view()


def create_phenix_data_loader(experiment_path: str) -> napari.Viewer:
    """
    Create an interactive Opera Phenix data loader widget in Napari.
    
    This function opens a Napari viewer with an interactive widget for selecting
    and visualizing different subsets of Opera Phenix imaging data.
    
    Parameters
    ----------
    experiment_path : str
        Path to the Opera Phenix experiment directory
        
    Returns
    -------
    viewer : napari.Viewer
        Napari viewer with the data loader widget docked
        
    Examples
    --------
    >>> viewer = create_phenix_data_loader('/path/to/experiment')
    """
    # Create viewer
    viewer = napari.Viewer()
    
    # Create and dock widget
    widget = PhenixDataLoaderWidget(experiment_path, viewer)
    viewer.window.add_dock_widget(widget, name="Data Loader", area="right")
    
    return viewer
