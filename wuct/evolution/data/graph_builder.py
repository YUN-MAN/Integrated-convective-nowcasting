"""
Build a directed graph from tracking objects and their properties.

This script processes rain cell tracking data to create a graph where:
- Nodes represent rain cells at each time step
- Edges represent temporal relationships between cells

Key features:
1. Outlier detection and removal based on property ratios
   - Ratios represent physical changes in rain cell properties between timesteps
   - Extremely large changes likely indicate erroneous data
   - Filtering removes unrealistic cell transitions
2. Calculation of similarity distances between consecutive nodes
   - Similarity distance is the sum of squared differences between normalized properties
   - Initially designed for selecting "most representative" paths in the graph
   - Currently not used in path selection (see all_path_DFS.py)
3. Graph filtering to remove unrealistic connections
   - Removes edges exceeding threshold values based on attribute ratios
   - Removes nodes with unrealistic properties
   - Maintains graph structure for further analysis

The resulting graph is saved for further analysis in single-core or multi-core path selection.

Usage:
    python build_graph.py --event_date YYYYMMDD

Input:
    - JSON files containing tracking objects
    - JSON file with all cell properties

Output:
    - GML file of the processed graph
"""

import os
import json
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Any, Optional
import numpy as np
import networkx as nx
import argparse

def load_json(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)

def get_node_key(time_str: str, obj: Dict) -> str:
    """Generate unique node key from time and object ID."""
    return f"{time_str}_{obj['fId']['lab']}_{obj['fId']['lev']}"

def get_node_attributes(properties: Dict, obj: Dict) -> Dict[str, float]:
    """
    Extract node attributes from properties and object data.
    
    Args:
        properties: Dictionary containing cell properties dataset
        obj: Dictionary containing object data from tracking_objects_ukng_1km.json
        
    Returns:
        Dictionary of node attributes including ttype, mean2d, smjr, smnr, area, altitude, and meanVIL
    """
    # TODO: minor updates 
    # after the properties dataset is updated (in historical_cell_extraction.py), 
    # the obj['ttype'] should be properties.get('ttype')
    # the obj['sumVals'] / len(obj['pts'][0]) should be properties.get('meanVIL')

    return {
        "ttype": obj['ttype'],
        "mean2d": properties.get('mean2d'),
        "smjr": properties.get('smjr'),
        "smnr": properties.get('smnr'),
        "area": properties.get('area'),
        "altitude": properties.get('centroid_z85'),
        "meanVIL": obj['sumVals'] / len(obj['pts'][0]) # VIL mean value
    }

def build_graph(folder: str, all_properties: Dict) -> Tuple[nx.DiGraph, Dict, Set, Set, Set]:
    """
    Construct a directed graph from tracking objects.

    Args:
        folder: Path to folder containing tracking object files
        all_properties: Dictionary of all cell properties

    Returns:
        Tuple containing:
        - NetworkX DiGraph object
        - Dictionary of node information
        - Set of newborn nodes
        - Set of merge nodes
        - Set of split nodes
    """
    G = nx.DiGraph(date='20180527')
    node_info = {}
    newborns, merges, splits = set(), set(), set()

    object_files = [os.path.join(folder, i) for i in os.listdir(folder) if 'tracking_objects_ukng_1km.json' in i]

    for object_file in object_files:
        current_time_str = os.path.basename(object_file).split('_')[0]
        current_datetime = datetime.strptime(current_time_str, '%Y%m%d%H%M')
        prev_time_str = (current_datetime - timedelta(minutes=5)).strftime('%Y%m%d%H%M')
        
        cell_jobj = load_json(object_file)
        
        for obj in cell_jobj['features']:
            current_node = f"{current_time_str}_{obj['fId']['lab']}_{obj['fId']['lev']}_{obj['ttype']}"
            node_key = get_node_key(current_time_str, obj)

            properties = all_properties.get(node_key, {})
            node_attributes = get_node_attributes(properties, obj)

            # check if all attributes are None
            for attr, value in node_attributes.items():
                if value is None:
                    print(f"Warning: {attr} is None for node {current_node}")

            # cell attributes are not None, add node to graph
            if all(value is not None for value in node_attributes.values()):
                node_info[current_node] = node_attributes
                G.add_node(current_node, **node_attributes)

                # add edges to graph
                if obj['ttype'] == 'I':
                    newborns.add(current_node)
                elif obj['ttype'] in ['C', 'M', 'S']:
                    prev_nodes = [f"{prev_time_str}_{pFId['lab']}_{pFId['lev']}" for pFId in obj['pFIds']]
                    for prev_node in prev_nodes:
                        prev_node = next((key for key in node_info if key.startswith(prev_node)), None)
                        if prev_node:
                            G.add_edge(prev_node, current_node)
                    
                    if obj['ttype'] == 'M':
                        merges.add(current_node)
                    elif obj['ttype'] == 'S':
                        splits.add(current_node)
            else:
                print(f"Skipping node {current_node} due to None attributes")

    return G, node_info, newborns, merges, splits

def calculate_attribute_ratios(G: nx.DiGraph, attributes: List[str]) -> Dict[str, List[float]]:
    """
    Calculate ratios of attributes between consecutive nodes.
    
    These ratios represent the physical changes in rain cell properties between 
    consecutive time steps. Large ratios (e.g., a cell suddenly growing to 10x its 
    previous size) often indicate erroneous data or tracking errors. These ratios 
    are used to filter out unrealistic cell transitions.
    
    Args:
        G: NetworkX DiGraph object
        attributes: List of attribute names to calculate ratios for
        
    Returns:
        Dictionary mapping attribute names to lists of their ratios
    """
    attribute_ratios = {attr: [] for attr in attributes}

    for prev_node, curr_node in G.edges():
        for attr in attributes:
            prev_value = G.nodes[prev_node].get(attr)
            curr_value = G.nodes[curr_node].get(attr)
            
            if prev_value and curr_value and prev_value != 0:
                ratio = curr_value / prev_value
                attribute_ratios[attr].append(ratio)

    return attribute_ratios

def calculate_statistics(attribute_ratios: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistical measures for attribute ratios.
    
    Args:
        attribute_ratios: Dictionary of attribute ratios
        
    Returns:
        Dictionary containing statistical measures for each attribute
    """
    statistics = {}
    for attr, ratios in attribute_ratios.items():
        if ratios:
            statistics[attr] = {
                'mean': np.mean(ratios),
                'median': np.median(ratios),
                '99th_percentile': np.percentile(ratios, 99),
                '99.9th_percentile': np.percentile(ratios, 99.9),
                'max': np.max(ratios),
                'min': np.min(ratios)
            }
    return statistics

def find_exceeding_edges(
    G: nx.DiGraph, 
    attribute_ratios: Dict[str, List[float]], 
    attributes: List[str], 
    percentile: float = 99
) -> Tuple[Dict[str, Set], Set, Set, Dict[str, float]]:
    """
    Find edges that exceed threshold values based on attribute ratios.
    
    Args:
        G: NetworkX DiGraph object
        attribute_ratios: Dictionary of attribute ratios
        attributes: List of attributes to check
        percentile: Percentile threshold for outlier detection
        
    Returns:
        Tuple containing:
        - Dictionary mapping attributes to sets of exceeding edges
        - Set of all exceeding edges
        - Set of all exceeding nodes
        - Dictionary of threshold values
    """
    thresholds = {attr: np.percentile(ratios, percentile) for attr, ratios in attribute_ratios.items()}
    exceeding_edges = defaultdict(set)
    all_exceeding_edges = set()
    all_exceeding_nodes = set()

    for prev_node, curr_node in G.edges():
        for attr in attributes:
            prev_value = G.nodes[prev_node].get(attr)
            curr_value = G.nodes[curr_node].get(attr)
            
            if prev_value and curr_value and prev_value != 0:
                ratio = curr_value / prev_value
                if ratio > thresholds[attr]:
                    exceeding_edges[attr].add((prev_node, curr_node))
                    all_exceeding_edges.add((prev_node, curr_node))
                    all_exceeding_nodes.add(curr_node)

    return exceeding_edges, all_exceeding_edges, all_exceeding_nodes, thresholds

def save_graph(G, filename):
    nx.write_gml(G, filename)

def get_all_nodes_properties_minmax(G: nx.DiGraph) -> nx.DiGraph:
    """
    Calculate min and max values for all node properties in the graph.
    
    Args:
        G: NetworkX DiGraph object
        
    Returns:
        Updated graph with min/max properties added
    """
    G.graph['min'] = {attr: float('inf') for attr in ['mean2d', 'smjr', 'smnr', 'area', 'altitude', 'meanVIL']}
    G.graph['max'] = {attr: float('-inf') for attr in ['mean2d', 'smjr', 'smnr', 'area', 'altitude', 'meanVIL']}

    for node in G.nodes():
        for attr, value in G.nodes[node].items():
            if attr in ['mean2d', 'smjr', 'smnr', 'area', 'altitude', 'meanVIL']:
                G.graph['min'][attr] = min(G.graph['min'][attr], value)
                G.graph['max'][attr] = max(G.graph['max'][attr], value)
    
    return G

def set_edge_attributes(G: nx.DiGraph) -> nx.DiGraph:
    """
    Calculate similarity distances between consecutive nodes.

    The similarity distance is calculated as the sum of squared differences between 
    normalized properties of consecutive nodes. While this metric was originally 
    designed for selecting "most representative" paths in the graph, it is currently 
    not used in the path selection process (see all_path_DFS.py). The function is 
    maintained for potential future use.
    
    Args:
        G: NetworkX DiGraph object
        
    Returns:
        Graph with edge similarity distances added
    """

    meanVIL_min, meanVIL_max = G.graph['min']['meanVIL'], G.graph['max']['meanVIL']
    smjr_min, smjr_max = G.graph['min']['smjr'], G.graph['max']['smjr']
    smnr_min, smnr_max = G.graph['min']['smnr'], G.graph['max']['smnr']

    if meanVIL_max == meanVIL_min:
        meanVIL_max = meanVIL_min + 1E-5
    if smjr_max == smjr_min:
        smjr_max = smjr_min + 1E-5
    if smnr_max == smnr_min:
        smnr_max = smnr_min + 1E-5

    meanVIL_Delta = meanVIL_max - meanVIL_min
    smjr_Delta = smjr_max - smjr_min
    smnr_Delta = smnr_max - smnr_min

    for prev_node, curr_node in G.edges():
        smjr_delta = abs((G.nodes[curr_node]['smjr'] - smjr_min) / smjr_Delta - (G.nodes[prev_node]['smjr'] - smjr_min) / smjr_Delta)
        smnr_delta = abs((G.nodes[curr_node]['smnr'] - smnr_min) / smnr_Delta - (G.nodes[prev_node]['smnr'] - smnr_min) / smnr_Delta)
        meanVIL_delta = abs((G.nodes[curr_node]['meanVIL'] - meanVIL_min) / meanVIL_Delta - (G.nodes[prev_node]['meanVIL'] - meanVIL_min) / meanVIL_Delta)

        similarity_dist = np.sum([smjr_delta**2, smnr_delta**2, meanVIL_delta**2])
        G[prev_node][curr_node]['similarity_dist'] = round(similarity_dist, 3)

    return G

def arg_parser() -> argparse.Namespace:
    """Configure and return command line argument parser."""
    parser = argparse.ArgumentParser(description="Build a graph from tracking objects and their properties.")
    parser.add_argument("--event_date", type=str, required=True, help="Event date in YYYYMMDD format")
    return parser.parse_args()

def main() -> None:
    """Main execution function."""
    
    args = arg_parser()
    event_date = args.event_date
    event_date_formatted = f"{event_date[:4]}-{event_date[4:6]}-{event_date[6:8]}"
    print(f"Processing event date: {event_date}")
    
    # Set load path and save path
    rootFolder = "/home/NAS/homes/yunman-10008/UK_tracking_output/YuShen_UKMO/MOCT/vfinal_vil3_kf_physical"
    folder = f'{rootFolder}/{event_date}/images'
    all_properties_file = f'/home/NAS/homes/yunman-10008/My-Research-Results/Analog_uncertainty/historical_cells/all_properties/all_properties_{event_date}.json'
    save_folder = f"/home/NAS/homes/yunman-10008/My-Research-Results/Event-evaluation/{event_date_formatted}/graph-model-paths/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Created folder: {save_folder}")
    save_graph_file = os.path.join(save_folder, f'rain_cells_graph_{event_date}.gml')

    # Load all properties
    all_properties = load_json(all_properties_file)

    # Build graph
    G, node_info, newborns, merges, splits = build_graph(folder, all_properties)

    # Calculate attribute ratios
    attributes = ['mean2d', 'smjr', 'smnr', 'area', 'altitude']
    attribute_ratios = calculate_attribute_ratios(G, attributes)
    statistics = calculate_statistics(attribute_ratios)

    # Find edges exceeding threshold values
    exceeding_edges, all_exceeding_edges, all_exceeding_nodes, thresholds = find_exceeding_edges(G, attribute_ratios, attributes)

    # Print statistics
    for attr, stats in statistics.items():
        print(f"Statistics for ratio of {attr.upper()}:")
        for stat_name, stat_value in stats.items():
            print(f"  {stat_name}: {stat_value:.3f}")
        print()

    # Print exceeding edges
    for attr, edges in exceeding_edges.items():
        print(f"Edges exceeding the 99th percentile threshold for {attr.upper()}:")
        print(f"Threshold: {thresholds[attr]}")
        print(f"Number of edges: {len(edges)}")
        print(f"Sample edges: {list(edges)[:2]}...")
        print()

    print(f"Total unique edges exceeding any threshold: {len(all_exceeding_edges)}")
    print(f"Total unique nodes exceeding any threshold: {len(all_exceeding_nodes)}")

    # Remove exceeding edges and nodes
    G_filtered = G.copy()
    G_filtered.remove_edges_from(all_exceeding_edges)
    G_filtered.remove_nodes_from(all_exceeding_nodes)

    print(f"Number of nodes in the original graph: {G.number_of_nodes()}")
    print(f"Number of edges in the original graph: {G.number_of_edges()}")
    print(f"Number of nodes in the filtered graph: {G_filtered.number_of_nodes()}")
    print(f"Number of edges in the filtered graph: {G_filtered.number_of_edges()}")

    # Calculate min and max values for all node properties in the filtered graph
    G_filtered = get_all_nodes_properties_minmax(G_filtered)

    # Calculate similarity distances
    G_filtered = set_edge_attributes(G_filtered)

    # Save the filtered graph
    save_graph(G_filtered, save_graph_file)
    print(f"Graph saved successfully in {save_graph_file}.")

if __name__ == "__main__":
    main()