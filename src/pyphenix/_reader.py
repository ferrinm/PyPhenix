import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import re
import ast
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
    timepoint_offsets: np.ndarray
    fields: List[int]
    planes: List[int]
    channel_ids: List[int]
    orientation_matrix: Optional[np.ndarray] = None

class FFCProfile:
    """Container for flat field correction profile"""
    def __init__(self, channel_id: int, ffc_data: dict):
        self.channel_id = channel_id
        self.character = ffc_data.get('Character', 'Null')
        self.mean = ffc_data.get('Mean', np.nan)
        self.quality = ffc_data.get('Quality', 0.0)
        
        profile = ffc_data.get('Profile', {})
        self.profile_type = profile.get('Type', 'Identity')
        
        if self.profile_type == 'Polynomial':
            self.coefficients = profile.get('Coefficients', [])
            self.dims = profile.get('Dims', [1080, 1080])
            self.origin = profile.get('Origin', [539.5, 539.5])
            self.scale = profile.get('Scale', [1.0, 1.0])
        else:
            self.coefficients = None
            self.dims = None
            self.origin = None
            self.scale = None
    
    def has_correction(self) -> bool:
        """Check if this profile has actual correction data"""
        return (self.profile_type == 'Polynomial' and 
                self.coefficients is not None and
                self.character == 'NonFlat')
    
    def generate_correction_image(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Generate the flat field correction image.

        Parameters
        ----------
        shape : tuple of int
            Output image shape (height, width) in (rows, cols) = (Y, X)

        Returns
        -------
        np.ndarray
            Correction multiplier image (1.0 = no correction)
        """
        if not self.has_correction():
            return np.ones(shape, dtype=np.float32)

        h, w = shape  # h = rows (Y), w = cols (X)

        # Create coordinate arrays
        # Opera Phenix uses image convention:
        #   origin[0] = Y center (row)
        #   origin[1] = X center (col)
        #   scale[0] = Y scale
        #   scale[1] = X scale
        y_coords = np.arange(h, dtype=np.float32) - self.origin[0]
        x_coords = np.arange(w, dtype=np.float32) - self.origin[1]

        # Scale coordinates
        y_coords = y_coords * self.scale[0]
        x_coords = x_coords * self.scale[1]

        # Create meshgrid
        # X[i,j] gives x-coordinate at row i, col j
        # Y[i,j] gives y-coordinate at row i, col j
        X, Y = np.meshgrid(x_coords, y_coords)

        # Evaluate polynomial: sum over orders n, then over terms within each order
        # SWAPPED: Try Y^k * X^(n-k) instead of X^k * Y^(n-k)
        # This accounts for row-major polynomial convention
        correction = np.zeros_like(X, dtype=np.float32)

        for n, coeffs in enumerate(self.coefficients):
            for k, coeff in enumerate(coeffs):
                # Swap: y^k * x^(n-k) instead of x^k * y^(n-k)
                correction += coeff * np.power(Y, k) * np.power(X, n - k)

        return correction

def parse_ffc_string(ffc_str: str) -> dict:
    """
    Parse the FFC profile string format.
    
    The string is formatted like:
    {Background: {Character: NonFlat, Mean: 153.32502, ...}, Channel: 1, ...}
    
    This is a custom format that needs careful parsing.
    """
    bg_result = {}  # Will hold the Background section data
    
    # Remove outer braces
    ffc_str = ffc_str.strip()
    if not ffc_str.startswith('{') or not ffc_str.endswith('}'):
        return {'Background': bg_result}
    
    # Find Background section
    bg_start = ffc_str.find('Background:')
    if bg_start == -1:
        return {'Background': bg_result}
    
    # Find the opening brace after 'Background:'
    brace_start = ffc_str.find('{', bg_start)
    if brace_start == -1:
        return {'Background': bg_result}
    
    # Find matching closing brace by counting
    brace_count = 0
    brace_end = -1
    for i in range(brace_start, len(ffc_str)):
        if ffc_str[i] == '{':
            brace_count += 1
        elif ffc_str[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                brace_end = i
                break
    
    if brace_end == -1:
        return {'Background': bg_result}
    
    # Extract Background content (everything between the braces)
    bg_content = ffc_str[brace_start+1:brace_end]
    
    # Extract Character
    char_match = re.search(r'Character:\s*(\w+)', bg_content)
    if char_match:
        bg_result['Character'] = char_match.group(1)
    
    # Extract Mean
    mean_match = re.search(r'Mean:\s*([\d.]+|NaN)', bg_content)
    if mean_match:
        mean_val = mean_match.group(1)
        bg_result['Mean'] = float(mean_val) if mean_val != 'NaN' else np.nan
    
    # Extract Quality
    quality_match = re.search(r'Quality:\s*([\d.]+)', bg_content)
    if quality_match:
        bg_result['Quality'] = float(quality_match.group(1))
    
    # Extract NoiseConst if present
    noise_match = re.search(r'NoiseConst:\s*([\d.]+)', bg_content)
    if noise_match:
        bg_result['NoiseConst'] = float(noise_match.group(1))
    
    # Extract Profile section using the same brace-counting method
    profile_start = bg_content.find('Profile:')
    if profile_start != -1:
        # Find the opening brace after 'Profile:'
        profile_brace_start = bg_content.find('{', profile_start)
        if profile_brace_start != -1:
            # Find matching closing brace
            brace_count = 0
            profile_brace_end = -1
            for i in range(profile_brace_start, len(bg_content)):
                if bg_content[i] == '{':
                    brace_count += 1
                elif bg_content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        profile_brace_end = i
                        break
            
            if profile_brace_end != -1:
                profile_content = bg_content[profile_brace_start+1:profile_brace_end]
                profile_dict = {}
                
                # Type
                type_match = re.search(r'Type:\s*(\w+)', profile_content)
                if type_match:
                    profile_dict['Type'] = type_match.group(1)
                
                # For Polynomial type, extract the arrays
                if profile_dict.get('Type') == 'Polynomial':
                    # Coefficients - use a more flexible pattern
                    # Match everything between Coefficients: and the next comma followed by a capital letter key
                    coeff_match = re.search(r'Coefficients:\s*(\[\[.*?\]\])(?=\s*,\s*[A-Z])', profile_content, re.DOTALL)
                    if coeff_match:
                        try:
                            coeff_str = coeff_match.group(1)
                            profile_dict['Coefficients'] = ast.literal_eval(coeff_str)
                        except Exception as e:
                            print(f"    ERROR parsing Coefficients: {e}")
                            print(f"    Coefficients string: {coeff_match.group(1)[:200]}")
                    else:
                        # Try alternate pattern - match until next key or end
                        coeff_match = re.search(r'Coefficients:\s*(\[\[[\d\s.,\-]+\]\])', profile_content, re.DOTALL)
                        if coeff_match:
                            try:
                                coeff_str = coeff_match.group(1)
                                profile_dict['Coefficients'] = ast.literal_eval(coeff_str)
                            except Exception as e:
                                print(f"    ERROR parsing Coefficients (alternate): {e}")
                    
                    # Dims
                    dims_match = re.search(r'Dims:\s*(\[[^\]]+\])', profile_content)
                    if dims_match:
                        try:
                            profile_dict['Dims'] = ast.literal_eval(dims_match.group(1))
                        except:
                            pass
                    
                    # Origin
                    origin_match = re.search(r'Origin:\s*(\[[^\]]+\])', profile_content)
                    if origin_match:
                        try:
                            profile_dict['Origin'] = ast.literal_eval(origin_match.group(1))
                        except:
                            pass
                    
                    # Scale
                    scale_match = re.search(r'Scale:\s*(\[[^\]]+\])', profile_content)
                    if scale_match:
                        try:
                            profile_dict['Scale'] = ast.literal_eval(scale_match.group(1))
                        except:
                            pass
                
                bg_result['Profile'] = profile_dict
    
    # Wrap in 'Background' key to match expected structure
    return {'Background': bg_result}

class OperaPhenixReader:
    """
    Reader for Opera Phenix exported experiment data.
    
    Reads TIFF images and XML metadata from exported experiments,
    returning numpy arrays and metadata dictionaries.
    """
    
    def __init__(self, experiment_path: str):
        """
        Initialize reader with path to experiment directory.

        Supports both:
        - Export format: Images/Index.xml
        - Archive format: images/ and index/ subdirectories
        
        Parameters
        ----------
        experiment_path : str
            Path to the exported experiment directory
        """
        self.experiment_path = Path(experiment_path)

        # Detect and set paths based on structure
        self._detect_structure()

        if not self.images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_path}")
        if not self.index_xml_path.exists():
            raise FileNotFoundError(f"Index XML not found: {self.index_xml_path}")

        # Parse metadata
        self.tree = ET.parse(self.index_xml_path)
        self.root = self.tree.getroot()
        self.ns = {'ns': '43B2A954-E3C3-47E1-B392-6635266B0DD3/HarmonyV7'}

        self.metadata = self._parse_metadata()
        self.image_index = self._build_image_index()
        self.well_field_map = self._build_well_field_map()
         
        # Load flat field correction profiles
        self.ffc_profiles = self._load_ffc_profiles()
    
    def _detect_ffc_path(self) -> Optional[Path]:
        """
        Detect the FFC profile XML file location.

        Returns
        -------
        Path or None
            Path to FFC XML file, or None if not found
        """
        # For export format: FFC_Profile/FFC_Profile_*.xml
        export_ffc_dir = self.experiment_path / "FFC_Profile"
        if export_ffc_dir.exists() and export_ffc_dir.is_dir():
            # Use iterdir() instead of glob() to handle spaces in filenames
            for item in export_ffc_dir.iterdir():
                if item.is_file() and item.suffix.lower() == '.xml':
                    return item

        # For archive format: flatfieldcorrection/*.xml
        archive_ffc_dir = self.experiment_path / "flatfieldcorrection"
        if archive_ffc_dir.exists() and archive_ffc_dir.is_dir():
            for item in archive_ffc_dir.iterdir():
                if item.is_file() and item.suffix.lower() == '.xml':
                    return item

        # Try case-insensitive directory search
        if self.experiment_path.exists():
            for item in self.experiment_path.iterdir():
                if item.is_dir():
                    # Check if directory name contains 'ffc' or 'flatfield'
                    name_lower = item.name.lower()
                    if 'ffc' in name_lower or 'flatfield' in name_lower:
                        # Look for XML files in this directory
                        for subitem in item.iterdir():
                            if subitem.is_file() and subitem.suffix.lower() == '.xml':
                                return subitem

        return None
    
    def _load_ffc_profiles(self) -> Dict[int, FFCProfile]:
        """
        Load flat field correction profiles from XML.

        Returns
        -------
        dict
            Dictionary mapping channel_id to FFCProfile
        """
        ffc_path = self._detect_ffc_path()
        if ffc_path is None:
            print("No flat field correction profiles found.")
            return {}

        print(f"Loading FFC profiles from: {ffc_path.name}")

        try:
            with open(ffc_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()

            # Parse XML
            tree = ET.parse(ffc_path)
            root = tree.getroot()

            # Parse namespace
            ns_match = re.match(r'\{(.+)\}', root.tag)
            if ns_match:
                ns = {'ns': ns_match.group(1)}
            else:
                ns = {}

            profiles = {}

            # Find all Map/Entry elements
            if ns:
                entries = root.findall('.//ns:Map/ns:Entry', ns)
            else:
                entries = root.findall('.//Map/Entry')

            print(f"  Found {len(entries)} entries in XML")

            for entry in entries:
                channel_id = int(entry.get('ChannelID'))
                print(f"\n  Processing Channel {channel_id}:")

                # Get FlatfieldProfile element
                if ns:
                    ffc_elem = entry.find('ns:FlatfieldProfile', ns)
                else:
                    ffc_elem = entry.find('FlatfieldProfile')

                if ffc_elem is not None and ffc_elem.text:
                    # Parse the custom format string
                    ffc_data = parse_ffc_string(ffc_elem.text)

                    # Extract Background section
                    if 'Background' in ffc_data:
                        bg_data = ffc_data['Background']
                        profiles[channel_id] = FFCProfile(channel_id, bg_data)
                        print(f"    Created FFCProfile")
                    else:
                        print(f"    WARNING: No Background section found in parsed data")
                else:
                    print(f"    WARNING: No FlatfieldProfile element or text found")

            # Print summary
            print(f"\n  Successfully loaded FFC profiles for {len(profiles)} channels:")
            for ch_id, profile in profiles.items():
                if profile.has_correction():
                    print(f"    Channel {ch_id}: {profile.character} "
                        f"(quality={profile.quality:.2f}, mean={profile.mean:.1f})")
                else:
                    print(f"    Channel {ch_id}: No correction (Identity)")

            return profiles

        except Exception as e:
            print(f"Warning: Could not load FFC profiles: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def apply_ffc(self, data: np.ndarray, channel_ids: List[int], 
                apply: bool = True) -> np.ndarray:
        """
        Apply flat field correction to image data.

        Parameters
        ----------
        data : np.ndarray
            Image data with shape (T, C, Z, Y, X)
        channel_ids : list of int
            List of channel IDs corresponding to C dimension
        apply : bool, default True
            Whether to actually apply correction (False = just return original)

        Returns
        -------
        np.ndarray
            Corrected image data (same shape as input, dtype=float32)
        """
        if not apply or not self.ffc_profiles:
            return data

        # Convert to float32 for correction
        corrected = data.astype(np.float32)

        for c_idx, ch_id in enumerate(channel_ids):
            if ch_id in self.ffc_profiles:
                profile = self.ffc_profiles[ch_id]

                if profile.has_correction():
                    # Generate illumination profile
                    img_shape = data.shape[-2:]  # (Y, X)
                    illumination = profile.generate_correction_image(img_shape)

                    # DIVIDE by illumination to correct vignetting
                    # This reduces bright center and brightens dark edges
                    corrected[:, c_idx, :, :, :] /= illumination[np.newaxis, np.newaxis, :, :]

                    print(f"  Applied FFC to channel {ch_id} (divided by illumination profile)")

        return corrected

    @staticmethod
    def row_to_letter(row_num: int) -> str:
        """
        Convert numeric row index to letter notation.
        
        Parameters
        ----------
        row_num : int
            1-based row number (1, 2, 3, ...)
        
        Returns
        -------
        str
            Letter notation ('A', 'B', 'C', ...)
        """
        return chr(ord('A') + row_num - 1)
    
    @staticmethod
    def letter_to_row(letter: str) -> int:
        """
        Convert letter row notation to numeric index.
        
        Parameters
        ----------
        letter : str
            Row letter ('A', 'B', 'C', ...)
        
        Returns
        -------
        int
            1-based row number (1, 2, 3, ...)
        """
        return ord(letter.upper()) - ord('A') + 1

    def _detect_structure(self):
        """
        Detect whether this is export or archive format and set paths accordingly.

        Sets:
        - self.structure_type: 'export', 'archive', or 'unknown'
        - self.images_path: path to images directory
        - self.index_xml_path: path to index XML file
        """
        # Try export format first (case-insensitive check)
        for images_name in ["Images", "images"]:
            export_images = self.experiment_path / images_name
            if export_images.exists():
                # Look for Index.xml (case variations)
                for index_name in ["Index.xml", "index.xml"]:
                    export_index = export_images / index_name
                    if export_index.exists():
                        self.structure_type = "export"
                        self.images_path = export_images
                        self.index_xml_path = export_index
                        print(f"Detected directory format: {images_name}/{index_name}")
                        return

        # Try archive format (images/ and separate index/ directory)
        archive_images = None
        for images_name in ["images", "Images"]:
            test_path = self.experiment_path / images_name
            if test_path.exists():
                archive_images = test_path
                break

        archive_index_dir = None
        for index_name in ["index", "Index"]:
            test_path = self.experiment_path / index_name
            if test_path.exists():
                archive_index_dir = test_path
                break

        if archive_images and archive_index_dir:
            # Find any XML file in the index directory
            index_xmls = list(archive_index_dir.glob("*.xml"))
            if index_xmls:
                self.structure_type = "archive"
                self.images_path = archive_images
                self.index_xml_path = index_xmls[0]  # Use first XML found
                print(f"Detected directory format: {archive_images.name}/ and {archive_index_dir.name}/")
                print(f"  Using index file: {self.index_xml_path.name}")
                return

        # Fallback: assume export format but warn
        print("WARNING: Could not definitively detect format. Assuming export format.")
        self.structure_type = "unknown"
        self.images_path = self.experiment_path / "Images"
        self.index_xml_path = self.images_path / "Index.xml"

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

    def _parse_orientation_matrix(self) -> Optional[np.ndarray]:
        """
        Parse the OrientationMatrix from channel metadata.

        Returns
        -------
        np.ndarray or None
            3x4 transformation matrix, or None if not found
        """
        channel_maps = self.root.findall('.//ns:Maps/ns:Map', self.ns)
        for map_elem in channel_maps:
            entries = map_elem.findall('ns:Entry', self.ns)
            for entry in entries:
                orient_elem = entry.find('ns:OrientationMatrix', self.ns)
                if orient_elem is not None:
                    # Parse string like "[[0.960371,0,0,-14.6],[0,-0.960371,0,4.4],[0,0,1.00,-0.033]]"
                    import re
                    import ast
                    matrix_str = orient_elem.text
                    matrix_list = ast.literal_eval(matrix_str)
                    return np.array(matrix_list)
        return None
    
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
        
        # Parse orientation matrix
        orientation_matrix = self._parse_orientation_matrix()

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
            channel_ids=sorted(channel_ids),
            orientation_matrix=orientation_matrix  # Add this field
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

        # Convert well IDs to letter notation
        wells_letter = []
        for well_id in self.metadata.wells:
            row_num, col_num = int(well_id[:2]), int(well_id[2:])
            row_letter = self.row_to_letter(row_num)
            wells_letter.append(f"{row_letter}{col_num:02d}")

        print(f"\nWells with data: {len(self.metadata.wells)}")
        print(f"  Wells: {', '.join(wells_letter)}")

        print(f"\nFields per well:")
        for well_id in self.metadata.wells:
            row, col = int(well_id[:2]), int(well_id[2:])
            row_letter = self.row_to_letter(row)
            fields = self.well_field_map.get((row, col), [])
            print(f"  {row_letter}{col:02d}: {len(fields)} fields ({min(fields) if fields else 'N/A'}-{max(fields) if fields else 'N/A'})")

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
                row: Optional[Union[int, str]] = None,
                column: Optional[int] = None,
                field: Optional[int] = None,
                stitch_fields: bool = False,
                timepoints: Optional[Union[int, List[int]]] = None,
                channels: Optional[Union[int, List[int]]] = None,
                z_slices: Optional[Union[int, List[int]]] = None,
                metadata_only: bool = False,
                apply_ffc: bool = True,
                output_file: Optional[str] = None,
                output_format: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
        """
        Read image data from Opera Phenix experiment.

        Parameters
        ----------
        row : int or str, optional
            Well row - either numeric (1, 2, 3...) or letter ('A', 'B', 'C'...)
            (default: first available)
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
        apply_ffc : bool, default True
            Whether to apply flat field correction if available
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

        # Convert row to numeric if letter provided
        if row is not None and isinstance(row, str):
            row = self.letter_to_row(row)

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
            # Pass apply_ffc parameter to stitching function
            data = self._read_and_stitch(row, column, fields, timepoints, 
                                        channels, z_slices, apply_ffc=apply_ffc)
            ffc_applied_during_stitch = apply_ffc and self.ffc_profiles
        else:
            data = self._read_images(row, column, fields, timepoints,
                                channels, z_slices)
            ffc_applied_during_stitch = False

        # Prepare metadata dictionary
        metadata_dict = self._prepare_metadata_dict(
            row, column, fields, timepoints, channels, z_slices, 
            stitch_fields, data.shape
        )

        # Print metadata
        self._print_metadata(metadata_dict)

        if not metadata_only and data is not None:
            # Apply flat field correction (only if not already applied during stitching)
            if apply_ffc and self.ffc_profiles and not ffc_applied_during_stitch:
                print("\nApplying flat field correction...")
                data = self.apply_ffc(data, channels, apply=True)
            elif ffc_applied_during_stitch:
                print("\nFlat field correction was applied during stitching.")
            elif not self.ffc_profiles:
                print("\nNo flat field correction profiles available.")
        
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
                        relative_url = self.image_index[key]['url']

                        # Construct path based on structure type
                        img_path = self._construct_image_path(relative_url, row, col)

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
    
    def _construct_image_path(self, relative_url: str, row: int, col: int) -> Path:
        """
        Construct full image path handling both export and archive formats.

        Archive format may use well subdirectories like r04c03/
        Export format has flat structure
        """
        if self.structure_type == "archive":
            # Try with well subdirectory first
            well_dir = f"r{row:02d}c{col:02d}"
            img_path = self.images_path / well_dir / relative_url

            if img_path.exists():
                return img_path

            # Fallback: try without well subdirectory
            img_path = self.images_path / relative_url
            if img_path.exists():
                return img_path

            # Return the well-subdirectory path for error reporting
            return self.images_path / well_dir / relative_url
        else:
            # Export format: flat structure
            return self.images_path / relative_url

    def _read_and_stitch(self, row: int, col: int, fields: List[int],
                        timepoints: List[int], channels: List[int],
                        z_slices: List[int], apply_ffc: bool = True) -> np.ndarray:
        """
        Read and stitch multiple fields.

        Parameters
        ----------
        row : int
            Well row number
        col : int
            Well column number
        fields : List[int]
            List of field IDs to stitch
        timepoints : List[int]
            List of timepoint IDs
        channels : List[int]
            List of channel IDs
        z_slices : List[int]
            List of z-slice IDs
        apply_ffc : bool, default True
            Whether to apply flat field correction before stitching

        Returns
        -------
        np.ndarray
            Stitched image data (T, C, Z, Y, X) as float32
        """
        # Get field positions
        field_positions = {}
        for field in fields:
            key = (row, col, field, z_slices[0], timepoints[0], channels[0])
            if key in self.image_index:
                pos = self.image_index[key]['position']

                # Apply inverse of orientation matrix scale to correct stage positions
                if self.metadata.orientation_matrix is not None:
                    matrix = self.metadata.orientation_matrix
                    sx = matrix[0, 0]  
                    sy = matrix[1, 1]  

                    pos_x_corrected = pos[0] / sx
                    pos_y_corrected = pos[1] / sy  # sy is negative, so this flips Y

                    field_positions[field] = (pos_x_corrected, pos_y_corrected)
                else:
                    field_positions[field] = (pos[0], pos[1])

        if not field_positions:
            raise ValueError(f"No valid field positions found for well r{row:02d}c{col:02d}")

        # Calculate stitched dimensions
        img_h, img_w = self.metadata.image_size
        pixel_size = self.metadata.pixel_size[0]

        positions_x = [pos[0] for pos in field_positions.values()]
        positions_y = [pos[1] for pos in field_positions.values()]

        min_x, max_x = min(positions_x), max(positions_x)
        min_y, max_y = min(positions_y), max(positions_y)

        # Use rounding for better precision
        stitched_w_float = (max_x - min_x) / pixel_size + img_w
        stitched_h_float = (max_y - min_y) / pixel_size + img_h

        stitched_w = int(np.round(stitched_w_float))
        stitched_h = int(np.round(stitched_h_float))

        # Debug output
        print(f"\nStitching debug info:")
        print(f"  Image size: {img_h}×{img_w} pixels")
        print(f"  Pixel size: {pixel_size*1e6:.4f} µm")
        print(f"  Field width: {img_w * pixel_size*1e6:.2f} µm")

        if self.metadata.orientation_matrix is not None:
            matrix = self.metadata.orientation_matrix
            print(f"  Orientation matrix:")
            print(f"    Original scale: ({matrix[0,0]:.6f}, {matrix[1,1]:.6f})")
            print(f"    Applied inverse scale: ({1/matrix[0,0]:.6f}, {1/abs(matrix[1,1]):.6f})")

        if len(field_positions) >= 2:
            positions = sorted(field_positions.items())
            field1_pos = positions[0][1]
            field2_pos = positions[1][1]
            dx = abs(field2_pos[0] - field1_pos[0]) * 1e6
            dy = abs(field2_pos[1] - field1_pos[1]) * 1e6

            # Calculate spacing to nearest neighbor (not maximum)
            spacing = min(dx, dy) if min(dx, dy) > 0 else max(dx, dy)

            print(f"  Field spacing (after correction): Δx={dx:.2f} µm, Δy={dy:.2f} µm")
            print(f"  Nearest neighbor spacing: {spacing:.2f} µm")
            print(f"  Expected overlap: {100 * (1 - spacing/(img_w * pixel_size*1e6)):.1f}%")

        print(f"  Position range X: {min_x:.6f} to {max_x:.6f} m")
        print(f"  Position range Y: {min_y:.6f} to {max_y:.6f} m")
        print(f"  Stitched size: {stitched_h} × {stitched_w} pixels")
        print(f"  Field positions:")

        # Initialize stitched array - USE FLOAT32 from the start
        n_time = len(timepoints)
        n_channels = len(channels)
        n_z = len(z_slices)

        data = np.zeros((n_time, n_channels, n_z, stitched_h, stitched_w),
                    dtype=np.float32)  # Changed from uint16 to float32

        # Track missing images
        missing_images = []

        # Generate FFC correction images once per channel (ONLY if apply_ffc is True)
        ffc_corrections = {}
        if apply_ffc and self.ffc_profiles:
            print(f"  Generating FFC corrections for stitching...")
            for ch_idx, ch_id in enumerate(channels):
                if ch_id in self.ffc_profiles:
                    profile = self.ffc_profiles[ch_id]
                    if profile.has_correction():
                        ffc_corrections[ch_id] = profile.generate_correction_image((img_h, img_w))
                        print(f"    Channel {ch_id}: FFC will be applied")
        elif not apply_ffc:
            print(f"  FFC disabled by user - no corrections will be applied")
        elif not self.ffc_profiles:
            print(f"  No FFC profiles available")

        # Stitch images with maximum intensity projection for overlaps
        for t_idx, timepoint in enumerate(timepoints):
            for c_idx, channel in enumerate(channels):
                for z_idx, z_slice in enumerate(z_slices):
                    for field in fields:
                        key = (row, col, field, z_slice, timepoint, channel)
                        if key in self.image_index:
                            relative_url = self.image_index[key]['url']
                            img_path = self._construct_image_path(relative_url, row, col)

                            if img_path.exists():
                                # Load image as float32
                                img = np.array(Image.open(img_path), dtype=np.float32)

                                # Apply FFC to this field BEFORE stitching (only if enabled)
                                if channel in ffc_corrections:
                                    img /= ffc_corrections[channel]

                                # Calculate position in stitched image with rounding
                                pos = field_positions[field]

                                x_offset = int(np.round((pos[0] - min_x) / pixel_size))

                                # Use min_y because sy is negative (already flipped by division)
                                y_offset = int(np.round((pos[1] - min_y) / pixel_size))

                                # Ensure offsets are within bounds
                                x_offset = max(0, min(x_offset, stitched_w - img_w))
                                y_offset = max(0, min(y_offset, stitched_h - img_h))

                                # Debug output for first timepoint/channel/z only
                                if t_idx == 0 and c_idx == 0 and z_idx == 0:
                                    print(f"    Field {field}: corrected stage ({pos[0]:.6f}, {pos[1]:.6f}) m "
                                        f"→ pixel offset ({x_offset}, {y_offset})")

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

        # Convert row to letter notation
        row_letter = self.row_to_letter(row)

        metadata = {
            'plate_id': self.metadata.plate_id,
            'plate_layout': {
                'rows': self.metadata.plate_rows,
                'columns': self.metadata.plate_columns
            },
            'well': f"{row_letter}{col:02d}",
            'well_numeric': f"r{row:02d}c{col:02d}",  # Keep numeric for compatibility
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
