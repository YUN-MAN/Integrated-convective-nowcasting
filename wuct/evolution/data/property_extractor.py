# Strategy pattern for property extraction
from abc import ABC, abstractmethod
import networkx as nx
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Set

from .utils.property_utils import (
    load_graph, load_json, get_unique_nodes,
    apportion_cell_properties
)

class BasePropertyExtractor(ABC):
    """Base class for property extraction using the strategy pattern."""
    
    def __init__(self, event_date: str):
        """
        Initialize the property extractor.
        
        Args:
            event_date (str): Event date in YYYYMMDD format
        """
        self.event_date = event_date
        self.event_date_formatted = f"{event_date[:4]}-{event_date[4:6]}-{event_date[6:]}"
        self.graph = None
        self.all_properties = None
        self.case_dict = None

    def load_data(self) -> None:
        """Load common data needed for all extractors."""
        graph_folder = f"/home/NAS/homes/yunman-10008/My-Research-Results/Event-evaluation/{self.event_date_formatted}/graph-model-paths"
        self.graph = nx.read_gml(os.path.join(graph_folder, f'rain_cells_graph_{self.event_date}.gml'))
        self.all_properties = self._load_json("/home/yunman/master_research/data_from_NAS/all_properties.json")
        
        import_folder = f"/home/NAS/homes/yunman-10008/My-Research-Results/Event-evaluation/{self.event_date_formatted}/graph-model-paths/DFS_all_paths"
        self.case_dict = self._load_json(os.path.join(import_folder, f'{self.case_type}_case.json'))

    @abstractmethod
    def process_properties(self) -> Dict:
        """
        Process properties for the specific case type.
        
        Returns:
            Dict: Processed properties in the format:
            {
                'case_id': {
                    'key': List[str],
                    'path': List[str],
                    'mean2d': List[float],
                    'smjr': List[float],
                    'smnr': List[float],
                    'area': List[float],
                    'altitude': List[float],
                    'topheight': List[int]
                }
            }
        """
        pass

    def save_properties(self, processed_props: Dict) -> None:
        """
        Save processed properties to JSON files.
        
        Args:
            processed_props (Dict): Dictionary of processed properties
        """
        save_folder = f"/home/NAS/homes/yunman-10008/My-Research-Results/Event-evaluation/{self.event_date_formatted}/DL/Input/{self.case_type}-case"
        os.makedirs(save_folder, exist_ok=True)

        # Separate properties into components
        property_files = self._organize_properties(processed_props)
        
        # Save each property type to its own file
        for prop_name, prop_dict in property_files.items():
            filename = f"all_{prop_name}_{self.case_type}.json"
            self._save_json(prop_dict, save_folder, filename)

    def _organize_properties(self, processed_props: Dict) -> Dict[str, Dict]:
        """
        Organize properties into components.

        Args:
            processed_props: Dictionary containing all properties for each case
                {
                    "case_id": {
                        "key": [...],
                        'path': [...],
                        'mean2d': [...],
                        'smjr': [...],
                        'smnr': [...],
                        'area': [...],
                        'altitude': [...],
                        'topheight': [...]
                    }
                }

        Returns:
            Dictionary of property dictionaries:
            {
                'key': {case_id: values},
                'path': {case_id: values},
                'mean2d': {case_id: values},
                ...
            }
        """
        property_files = {
            'key': {},
            'path': {},
            'mean2d': {},
            'smjr': {},
            'smnr': {},
            'area': {},
            'altitude': {},
            'topheight': {}
        }

        for case_id, values in processed_props.items():
            property_files['key'][case_id] = values['key']
            # Remove tracking type suffix for path
            property_files['path'][case_id] = [
                node[:-2] if len(node.split('_')) > 3 else node 
                for node in values['path']
            ]
            property_files['mean2d'][case_id] = values['mean2d']
            property_files['smjr'][case_id] = values['smjr']
            property_files['smnr'][case_id] = values['smnr']
            property_files['area'][case_id] = values['area']
            property_files['altitude'][case_id] = values['altitude']
            property_files['topheight'][case_id] = values['topheight']

        return property_files

    @staticmethod
    def _load_json(filepath: str) -> Dict:
        """Load JSON data from file."""
        with open(filepath, 'r') as f:
            return json.load(f)

    @staticmethod
    def _save_json(data: Dict, folder: str, filename: str) -> None:
        """Save data to JSON file."""
        with open(os.path.join(folder, filename), 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved {filename} to {folder}")

    @property
    @abstractmethod
    def case_type(self) -> str:
        """Return the type of case being processed."""
        pass

class ContinuePropertyExtractor(BasePropertyExtractor):
    """Extractor for continue cases where a cell continues without splitting or merging."""

    @property
    def case_type(self) -> str:
        return "continue"

    def process_properties(self) -> Dict:
        """
        Process properties for continue cases.
        
        For continue cases, we directly extract properties from all_properties
        without needing to apportion values between cells.
        
        Returns:
            Dict: Processed properties dictionary
        """
        processed_props = {}
        
        # Process each path in continue cases
        for key, path in self.case_dict.items():
            props = {
                "key": path,
                "path": path,
                "mean2d": [],
                "smjr": [],
                "smnr": [],
                "area": [],
                "altitude": [],
                "topheight": [],
            }
            
            # Process each node in the path
            for node in path:
                # Extract cell key without tracking type
                date_time, lab, lev, ttype = node.split("_")
                cell_key = f"{date_time}_{lab}_{lev}"
                
                if cell_key in self.all_properties:
                    cell_data = self.all_properties[cell_key]
                    
                    # Extract properties for this cell
                    props["mean2d"].append(float(cell_data["mean2d"]))
                    props["smjr"].append(float(cell_data["smjr"]))
                    props["smnr"].append(float(cell_data["smnr"]))
                    props["area"].append(float(cell_data["area"]))
                    props["altitude"].append(float(cell_data["centroid_z85"]))
                    props["topheight"].append(int(max(cell_data["pts_3d"][0])))
                else:
                    print(f"Warning: Cell key {cell_key} not found in all_properties")
                    continue
            
            # Only add to processed_props if we have data
            if len(props["mean2d"]) > 0:
                processed_props[key] = props
            else:
                print(f"Warning: No valid properties found for path {key}")
        
        # Rename keys to be consecutive integers as strings
        processed_props_renamed = {
            str(i): props for i, props in enumerate(processed_props.values())
        }
        
        print(f"Processed {len(processed_props_renamed)} continue cases")
        return processed_props_renamed

    def extract(self) -> None:
        """
        Main method to orchestrate the property extraction process.
        """
        try:
            # Load required data
            self.load_data()
            
            # Process the properties
            processed_props = self.process_properties()
            
            if processed_props:
                # Save the processed properties
                self.save_properties(processed_props)
                print("Successfully processed and saved continue case properties")
            else:
                print("No valid continue cases found to process")
                
        except Exception as e:
            print(f"Error processing continue cases: {str(e)}")
            raise

class MergePropertyExtractor(BasePropertyExtractor):
    """Extractor for merge cases where multiple cells merge into one."""

    @property
    def case_type(self) -> str:
        return "merge"

    def _create_merge_properties_lookup(self) -> Tuple[Dict, Set]:
        """
        Create a lookup dictionary containing apportioned properties for all merge nodes 
        and their sub-features (predecessors).
        
        Returns:
            Tuple[Dict, Set]: (merge_props_lookup, all_invalid_subfeatures)
                merge_props_lookup structure:
                {
                    'merge_node_id': {
                        'predecessor_0_id': {
                            'mean2d': mean2d_apportioned,
                            ...
                        }
                    }
                }
        """
        from .utils.property_utils import apportion_cell_properties, get_unique_nodes
        
        merge_props_lookup = dict()
        all_invalid_subfeatures = set()

        # Get all unique merge nodes (t nodes)
        all_merge_nodes = get_unique_nodes(self.case_dict, 3)

        for merge_node in all_merge_nodes:
            predecessors = list(self.graph.predecessors(merge_node))
            if len(predecessors) > 1:
                # Calculate apportioned properties for this merge node
                subfeatures_accounted, apportioned_properties = apportion_cell_properties(
                    self.all_properties,
                    merge_node,
                    predecessors
                )

                # Track invalid subfeatures
                subfeatures_invalid = set(predecessors) - set(subfeatures_accounted)
                if subfeatures_invalid:
                    print(f"Warning: {merge_node} has invalid subfeatures: {subfeatures_invalid}")
                    all_invalid_subfeatures.update(subfeatures_invalid)

                # Store valid apportioned properties
                merge_props_lookup[merge_node] = {}
                for subfeature in subfeatures_accounted:
                    merge_props_lookup[merge_node][subfeature] = {
                        'mean2d': apportioned_properties[subfeature]['mean2d'],
                        'smjr': apportioned_properties[subfeature]['smjr'],
                        'smnr': apportioned_properties[subfeature]['smnr'],
                        'area': apportioned_properties[subfeature]['area'],
                        'altitude': apportioned_properties[subfeature]['altitude'],
                        'topheight': apportioned_properties[subfeature]['topheight'],
                    }
            else:
                print(f"Warning: {merge_node} has only one predecessor. Invalid merge node.")

        return merge_props_lookup, all_invalid_subfeatures

    def _filter_invalid_merge_cases(self, all_invalid_subfeatures: Set) -> Dict:
        """
        Remove merge cases where the merged node has invalid subfeatures (predecessors).
        
        Args:
            all_invalid_subfeatures (Set): Set of invalid subfeature IDs
            
        Returns:
            Dict: Filtered merge cases
        """
        print(f"Filtering merge cases with invalid predecessor paths...")
        print(f"Length before filter: {len(self.case_dict)}")
        
        filtered_merge_cases = {}
        for key, path in self.case_dict.items():
            merge_subfeature = path[2]  # the merged node is the third (t-5) node
            if merge_subfeature not in all_invalid_subfeatures:
                filtered_merge_cases[key] = path
        
        print(f"Removed {len(self.case_dict) - len(filtered_merge_cases)} invalid merge cases")
        print(f"Length after filter: {len(filtered_merge_cases)}")
        return filtered_merge_cases

    def process_properties(self) -> Dict:
        """
        Process properties for merge cases.
        
        This method:
        1. Creates a lookup for apportioned properties
        2. Filters out invalid cases
        3. Processes each valid merge path
        
        Returns:
            Dict: Processed properties dictionary
        """
        # Create lookup for merge properties
        merge_props_lookup, all_invalid_subfeatures = self._create_merge_properties_lookup()
        
        # Filter out invalid cases
        merge_cases_filtered = self._filter_invalid_merge_cases(all_invalid_subfeatures)
        
        processed_props = {}
        
        # Process each valid merge path
        for case_key, path in merge_cases_filtered.items():
            merge_subfeature = path[2]  # t-5 node
            merge_node = path[3]  # t node

            if merge_node in merge_props_lookup and merge_subfeature in merge_props_lookup[merge_node]:
                processed_props[case_key] = {
                    'key': path,
                    'path': path,
                    'mean2d': [],
                    'smjr': [],
                    'smnr': [],
                    'area': [],
                    'altitude': [],
                    'topheight': [],
                }

                # Add t-15, t-10, t-5 properties
                for pred_node in path[:3]:
                    node_key = pred_node[:-2]  # remove tracking type suffix
                    if node_key in self.all_properties:
                        cell_data = self.all_properties[node_key]
                        processed_props[case_key]['mean2d'].append(float(cell_data['mean2d']))
                        processed_props[case_key]['smjr'].append(float(cell_data['smjr']))
                        processed_props[case_key]['smnr'].append(float(cell_data['smnr']))
                        processed_props[case_key]['area'].append(float(cell_data['area']))
                        processed_props[case_key]['altitude'].append(float(cell_data['centroid_z85']))
                        processed_props[case_key]['topheight'].append(int(max(cell_data['pts_3d'][0])))

                # Add apportioned properties for the merged node (at t)
                merge_props = merge_props_lookup[merge_node][merge_subfeature]
                processed_props[case_key]['mean2d'].append(float(merge_props['mean2d']))
                processed_props[case_key]['smjr'].append(float(merge_props['smjr']))
                processed_props[case_key]['smnr'].append(float(merge_props['smnr']))
                processed_props[case_key]['area'].append(float(merge_props['area']))
                processed_props[case_key]['altitude'].append(float(merge_props['altitude']))
                processed_props[case_key]['topheight'].append(float(merge_props['topheight']))

                # Add successors properties (t+5, t+10, t+15)
                for succ_node in path[4:]:
                    node_key = succ_node[:-2]  # remove tracking type suffix
                    if node_key in self.all_properties:
                        cell_data = self.all_properties[node_key]
                        processed_props[case_key]['mean2d'].append(float(cell_data['mean2d']))
                        processed_props[case_key]['smjr'].append(float(cell_data['smjr']))
                        processed_props[case_key]['smnr'].append(float(cell_data['smnr']))
                        processed_props[case_key]['area'].append(float(cell_data['area']))
                        processed_props[case_key]['altitude'].append(float(cell_data['centroid_z85']))
                        processed_props[case_key]['topheight'].append(int(max(cell_data['pts_3d'][0])))

        # Rename keys to be consecutive integers as strings
        processed_props_renamed = {
            str(i): props for i, props in enumerate(processed_props.values())
        }
        
        print(f"Processed {len(processed_props_renamed)} merge cases")
        return processed_props_renamed

    def extract(self) -> None:
        """
        Main method to orchestrate the merge property extraction process.
        """
        try:
            # Load required data
            self.load_data()
            
            # Process the properties
            processed_props = self.process_properties()
            
            if processed_props:
                # Save the processed properties
                self.save_properties(processed_props)
                print("Successfully processed and saved merge case properties")
            else:
                print("No valid merge cases found to process")
                
        except Exception as e:
            print(f"Error processing merge cases: {str(e)}")
            raise

class SplitPropertyExtractor(BasePropertyExtractor):
    """Extractor for split cases where one cell splits into multiple cells."""

    @property
    def case_type(self) -> str:
        return "split"

    def _create_split_properties_lookup(self) -> Tuple[Dict, Set]:
        """
        Create a lookup dictionary containing apportioned properties for all split source nodes 
        and their sub-features (successors).
        
        Returns:
            Tuple[Dict, Set]: (split_props_lookup, all_invalid_subfeatures)
                split_props_lookup structure:
                {
                    'split_source_node_id': {
                        'successor_0_id': {
                            'mean2d': mean2d_apportioned,
                            ...
                        }
                    }
                }
        """
        from .utils.property_utils import apportion_cell_properties, get_unique_nodes
        
        split_props_lookup = dict()
        all_invalid_subfeatures = set()

        # Get all unique split source nodes (t nodes)
        all_split_sources = get_unique_nodes(self.case_dict, 2)

        for split_source_node in all_split_sources:
            successors = list(self.graph.successors(split_source_node))
            if len(successors) > 1:
                # Calculate apportioned properties for this split source node
                subfeatures_accounted, apportioned_properties = apportion_cell_properties(
                    self.all_properties,
                    split_source_node,
                    successors
                )

                # Track invalid subfeatures
                subfeatures_invalid = set(successors) - set(subfeatures_accounted)
                if subfeatures_invalid:
                    print(f"Warning: {split_source_node} has invalid subfeatures: {subfeatures_invalid}")
                    all_invalid_subfeatures.update(subfeatures_invalid)

                # Store valid apportioned properties
                split_props_lookup[split_source_node] = {}
                for subfeature in subfeatures_accounted:
                    split_props_lookup[split_source_node][subfeature] = {
                        'mean2d': apportioned_properties[subfeature]['mean2d'],
                        'smjr': apportioned_properties[subfeature]['smjr'],
                        'smnr': apportioned_properties[subfeature]['smnr'],
                        'area': apportioned_properties[subfeature]['area'],
                        'altitude': apportioned_properties[subfeature]['altitude'],
                        'topheight': apportioned_properties[subfeature]['topheight'],
                    }
            else:
                print(f"Warning: {split_source_node} has only one successor. Invalid split source node.")

        return split_props_lookup, all_invalid_subfeatures

    def _filter_invalid_split_cases(self, all_invalid_subfeatures: Set) -> Dict:
        """
        Remove split cases where the split source node has invalid subfeatures (successors).
        
        Args:
            all_invalid_subfeatures (Set): Set of invalid subfeature IDs
            
        Returns:
            Dict: Filtered split cases
        """
        print(f"Filtering split cases with invalid successor paths...")
        print(f"Length before filter: {len(self.case_dict)}")
        
        filtered_split_cases = {}
        for key, path in self.case_dict.items():
            split_source_node = path[3]  # the split node at time t is the fourth node
            if split_source_node not in all_invalid_subfeatures:
                filtered_split_cases[key] = path
        
        print(f"Removed {len(self.case_dict) - len(filtered_split_cases)} invalid split cases")
        print(f"Length after filter: {len(filtered_split_cases)}")
        return filtered_split_cases

    def process_properties(self) -> Dict:
        """
        Process properties for split cases.
        
        This method:
        1. Creates a lookup for apportioned properties
        2. Filters out invalid cases
        3. Processes each valid split path
        
        Returns:
            Dict: Processed properties dictionary
        """
        # Create lookup for split properties
        split_props_lookup, all_invalid_subfeatures = self._create_split_properties_lookup()
        
        # Filter out invalid cases
        split_cases_filtered = self._filter_invalid_split_cases(all_invalid_subfeatures)
        
        processed_props = {}
        
        # Process each valid split path
        for case_key, path in split_cases_filtered.items():
            split_source_node = path[2]  # t-5 node
            split_node = path[3]  # t node

            if split_source_node in split_props_lookup and split_node in split_props_lookup[split_source_node]:
                processed_props[case_key] = {
                    'key': path,
                    'path': path,
                    'mean2d': [],
                    'smjr': [],
                    'smnr': [],
                    'area': [],
                    'altitude': [],
                    'topheight': [],
                }

                # Add t-15, t-10, t-5 properties
                for pred_node in path[:3]:
                    node_key = pred_node[:-2]  # remove tracking type suffix
                    if node_key in self.all_properties:
                        cell_data = self.all_properties[node_key]
                        processed_props[case_key]['mean2d'].append(float(cell_data['mean2d']))
                        processed_props[case_key]['smjr'].append(float(cell_data['smjr']))
                        processed_props[case_key]['smnr'].append(float(cell_data['smnr']))
                        processed_props[case_key]['area'].append(float(cell_data['area']))
                        processed_props[case_key]['altitude'].append(float(cell_data['centroid_z85']))
                        processed_props[case_key]['topheight'].append(int(max(cell_data['pts_3d'][0])))

                # Add apportioned split source properties
                split_props = split_props_lookup[split_source_node][split_node]
                processed_props[case_key]['mean2d'].append(float(split_props['mean2d']))
                processed_props[case_key]['smjr'].append(float(split_props['smjr']))
                processed_props[case_key]['smnr'].append(float(split_props['smnr']))
                processed_props[case_key]['area'].append(float(split_props['area']))
                processed_props[case_key]['altitude'].append(float(split_props['altitude']))
                processed_props[case_key]['topheight'].append(float(split_props['topheight']))

                # Add successor properties (t to t+15)
                for succ_node in path[4:]:
                    node_key = succ_node[:-2]  # remove tracking type suffix
                    if node_key in self.all_properties:
                        cell_data = self.all_properties[node_key]
                        processed_props[case_key]['mean2d'].append(float(cell_data['mean2d']))
                        processed_props[case_key]['smjr'].append(float(cell_data['smjr']))
                        processed_props[case_key]['smnr'].append(float(cell_data['smnr']))
                        processed_props[case_key]['area'].append(float(cell_data['area']))
                        processed_props[case_key]['altitude'].append(float(cell_data['centroid_z85']))
                        processed_props[case_key]['topheight'].append(int(max(cell_data['pts_3d'][0])))

        # Rename keys to be consecutive integers as strings
        processed_props_renamed = {
            str(i): props for i, props in enumerate(processed_props.values())
        }
        
        print(f"Processed {len(processed_props_renamed)} split cases")
        return processed_props_renamed

    def extract(self) -> None:
        """
        Main method to orchestrate the split property extraction process.
        """
        try:
            # Load required data
            self.load_data()
            
            # Process the properties
            processed_props = self.process_properties()
            
            if processed_props:
                # Save the processed properties
                self.save_properties(processed_props)
                print("Successfully processed and saved split case properties")
            else:
                print("No valid split cases found to process")
                
        except Exception as e:
            print(f"Error processing split cases: {str(e)}")
            raise

class AllCasePropertyExtractor(BasePropertyExtractor):
    """Extractor for combining all case types (continue, merge, split) into a single dataset."""

    @property
    def case_type(self) -> str:
        return "all"

    def _check_case_data(self, case_type: str) -> bool:
        """
        Check if case data exists for the given case type.
        
        Args:
            case_type (str): Type of case to check ('continue', 'merge', or 'split')
            
        Returns:
            bool: True if case data exists, False otherwise
        """
        case_path = f"/home/NAS/homes/yunman-10008/My-Research-Results/Event-evaluation/{self.event_date_formatted}/DL/Input/{case_type}-case/all_path_{case_type}.json"
        try:
            with open(case_path, "r") as f:
                case_dict = json.load(f)
            if len(case_dict) == 0:
                print(f"The case data of {case_type} does not exist for {self.event_date_formatted}")
                return False
            return True
        except FileNotFoundError:
            print(f"Case file not found for {case_type}")
            return False

    def _load_case_properties(self, case_type: str) -> Dict:
        """
        Load properties for a specific case type.
        
        Args:
            case_type (str): Type of case to load ('continue', 'merge', or 'split')
            
        Returns:
            Dict: Dictionary containing all properties for the case type
        """
        input_folder = f"/home/NAS/homes/yunman-10008/My-Research-Results/Event-evaluation/{self.event_date_formatted}/DL/Input/{case_type}-case"
        case_properties = {}
        
        # Load each property type
        for prop in ['key', 'path', 'mean2d', 'smjr', 'smnr', 'area', 'altitude', 'topheight']:
            prop_file = os.path.join(input_folder, f'all_{prop}_{case_type}.json')
            try:
                case_properties[prop] = self._load_json(prop_file)
            except FileNotFoundError:
                print(f"Warning: Property file {prop_file} not found")
                return None
        
        return case_properties

    def _combine_case_properties(self, case_properties: Dict[str, Dict]) -> Dict:
        """
        Combine properties from multiple cases into a single dictionary.
        
        Args:
            case_properties: Dictionary mapping case types to their properties
                {
                    'continue': {prop_dict},
                    'merge': {prop_dict},
                    'split': {prop_dict}
                }
                
        Returns:
            Dict: Combined properties dictionary
        """
        combined_properties = {}
        counter = 0
        
        # Initialize property dictionaries
        property_types = ['key', 'path', 'mean2d', 'smjr', 'smnr', 'area', 'altitude', 'topheight']
        for prop in property_types:
            combined_properties[prop] = {}
        
        # Combine properties from each case type
        for case_type, properties in case_properties.items():
            if not properties:
                continue
                
            path_keys = properties['path'].keys()
            print(f"Adding {len(path_keys)} entries from {case_type} case")
            
            for key in path_keys:
                # Add all properties with a new consecutive counter
                for prop in property_types:
                    combined_properties[prop][str(counter)] = properties[prop][key]
                counter += 1
        
        return combined_properties

    def process_properties(self) -> Dict:
        """
        Process and combine properties from all case types.
        
        Returns:
            Dict: Combined properties dictionary
        """
        case_types = ['continue', 'merge', 'split']
        case_properties = {}
        
        # Load properties for each case type
        for case_type in case_types:
            if self._check_case_data(case_type):
                case_properties[case_type] = self._load_case_properties(case_type)
                if case_properties[case_type]:
                    print(f"Loaded {len(case_properties[case_type]['path'])} {case_type} cases")
        
        # Combine all case properties
        combined_properties = self._combine_case_properties(case_properties)
        
        print(f"Combined {len(combined_properties['path'])} total cases")
        return combined_properties

    def extract(self) -> None:
        """
        Main method to orchestrate the all-case property extraction process.
        """
        try:
            # Process and combine properties
            processed_props = self.process_properties()
            
            if processed_props:
                # Save the combined properties
                self.save_properties(processed_props)
                print("Successfully processed and saved all case properties")
            else:
                print("No valid cases found to process")
                
        except Exception as e:
            print(f"Error processing all cases: {str(e)}")
            raise

    def load_data(self) -> None:
        """
        Override load_data as it's not needed for all-case processing.
        All-case extractor loads pre-processed data from individual case files.
        """
        pass

class PropertyExtractor:
    """Main class for orchestrating property extraction across all case types."""
    
    def __init__(self):
        """Initialize extractors for each case type."""
        self.extractors = {
            'continue': ContinuePropertyExtractor,
            'merge': MergePropertyExtractor,
            'split': SplitPropertyExtractor,
            'all': AllCasePropertyExtractor
        }
    
    def extract(self, case_type: str, event_date: str) -> None:
        """
        Extract properties for the specified case type.
        
        Args:
            case_type (str): Type of case to process ('continue', 'merge', 'split', or 'all')
            event_date (str): Event date in YYYYMMDD format
        
        Raises:
            ValueError: If case_type is invalid
        """
        if case_type not in self.extractors:
            raise ValueError(f"Unknown case type: {case_type}. Valid types are: {list(self.extractors.keys())}")
        
        # Create and run the appropriate extractor
        extractor = self.extractors[case_type](event_date)
        extractor.extract()
    
    def extract_all_types(self, event_date: str) -> None:
        """
        Extract properties for all case types in sequence.
        
        Args:
            event_date (str): Event date in YYYYMMDD format
        """
        # Process individual cases first
        for case_type in ['continue', 'merge', 'split']:
            print(f"\nProcessing {case_type} cases...")
            try:
                self.extract(case_type, event_date)
            except Exception as e:
                print(f"Error processing {case_type} cases: {str(e)}")
        
        # Process combined cases last
        print("\nProcessing combined cases...")
        try:
            self.extract('all', event_date)
        except Exception as e:
            print(f"Error processing combined cases: {str(e)}")

def main():
    """Example usage of the PropertyExtractor system."""
    
    # Create the main extractor
    extractor = PropertyExtractor()
    
    # Example event date
    event_date = "20230101"  # YYYYMMDD format
    
    # Method 1: Extract a specific case type
    print("Extracting continue cases...")
    extractor.extract('continue', event_date)
    
    # Method 2: Extract all case types in sequence
    print("\nExtracting all case types...")
    extractor.extract_all_types(event_date)

if __name__ == "__main__":
    # Example command-line interface
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract properties for rain cell tracking cases.")
    parser.add_argument("--event_date", type=str, required=True, help="Event date in YYYYMMDD format")
    parser.add_argument("--case_type", type=str, default="all", 
                       choices=['continue', 'merge', 'split', 'all', 'all_types'],
                       help="Type of case to process")
    
    args = parser.parse_args()
    
    # Create and run extractor
    extractor = PropertyExtractor()
    
    if args.case_type == 'all_types':
        extractor.extract_all_types(args.event_date)
    else:
        extractor.extract(args.case_type, args.event_date)

"""
1. Process a single case type:
    python property_extractor.py --event_date 20230101 --case_type continue

2. Process all case types sequentially:
    python property_extractor.py --event_date 20230101 --case_type all_types

3. Or use it programmatically:

    from property_extractor import PropertyExtractor

    # Create extractor
    extractor = PropertyExtractor()

    # Process specific case type
    extractor.extract('merge', '20230101')

    # Or process all types
    extractor.extract_all_types('20230101')

"""