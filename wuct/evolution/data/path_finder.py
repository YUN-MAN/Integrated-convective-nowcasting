"""
Find and classify all possible paths in a rain cell tracking graph.

This script performs the following operations:
1. Uses Depth First Search (DFS) to find all paths in the graph.
2. Creates sliding windows of size 7 (t-15 to t+15 minutes).
3. Classifies each window as 'continue', 'merge', 'split', or 'unknown'.
4. Saves results in JSON format.

Usage:
    python all_path_DFS.py --event_date YYYYMMDD

Input:
    - GML file of the rain cell tracking graph

Output:
    JSON files containing:
    - all_paths.json: Complete paths through the graph
    - all_windows.json: Sliding windows of size 7
    - continue_case.json: Windows showing continuous movement
    - merge_case.json: Windows showing cell merging
    - split_case.json: Windows showing cell splitting
"""

import os
import json
import networkx as nx
from typing import List, Dict, Set, Tuple, Any, Optional
from datetime import datetime
import argparse

def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find and classify all possible paths in a rain cell tracking graph."
    )
    parser.add_argument(
        "--event_date",
        type=str,
        required=True,
        help="Event date in YYYYMMDD format"
    )
    return parser.parse_args()

# Initialize global constants
args = arg_parser()
EVENT_DATE = args.event_date  # format YYYYMMDD
EVENT_DATE_FORMATTED = (
    f"{EVENT_DATE[:4]}-{EVENT_DATE[4:6]}-{EVENT_DATE[6:8]}"  # format YYYY-MM-DD
)
WINDOW_SIZE = 7 # t-15 to t+15 minutes
SAVE_FOLDER = (
    f"/home/NAS/homes/yunman-10008/My-Research-Results/Event-evaluation/"
    f"{EVENT_DATE_FORMATTED}/graph-model-paths/"
)
DFS_SAVE_FOLDER = os.path.join(SAVE_FOLDER, "DFS_all_paths")
GRAPH_FILE = os.path.join(SAVE_FOLDER, f'rain_cells_graph_{EVENT_DATE}.gml')


def load_graph(filename: str) -> nx.Graph:
    return nx.read_gml(filename)

def get_source_nodes(graph: nx.Graph) -> List[str]:
    """
    Find all source nodes in the graph (nodes with no incoming edges).

    Args:
        graph (nx.Graph): The rain cell tracking graph
        
    Returns:
        List[str]: List of source node identifiers
    """
    return [
        node for node in graph.nodes() 
        if graph.in_degree(node) == 0 and graph.out_degree(node) > 0
    ]

def dfs_paths(graph: nx.Graph, start_node: str) -> List[List[str]]:
    """
    Perform Depth First Search to find all possible paths from a start node.

    Args:
        graph (nx.Graph): The rain cell tracking graph
        start_node (str): Starting node for path finding

    Returns:
        List[List[str]]: List of all possible paths from the start node
    """
    paths = []
    stack = [(start_node, [start_node])]

    while stack:
        vertex, path = stack.pop()
        # Get unvisited neighbors
        neighbors: Set[str] = set(graph.neighbors(vertex)) - set(path)
        for next_node in neighbors:
            if not list(graph.successors(next_node)):  # End node reached
                paths.append(path + [next_node])
            else:
                stack.append((next_node, path + [next_node]))
    return paths

def is_valid_node(node: str) -> bool:
    """
    Check if a node represents a valid time point, 
    not close to start (00:00) and end (23:55) time.
    """
    time_str = node.split('_')[0]
    time = datetime.strptime(time_str, '%Y%m%d%H%M').time()
    return not (
        time == datetime.strptime("00:00", "%H:%M").time()
        or time == datetime.strptime("23:55", "%H:%M").time()
    )

def create_sliding_windows(
    paths: List[List[str]], 
    window_size: int = WINDOW_SIZE
) -> List[List[str]]:
    """
    Create sliding windows of specified size from paths.

    Args:
        paths (List[List[str]]): List of complete paths
        window_size (int): Size of the sliding window

    Returns:
        List[List[str]]: List of all windows
    """
    return [
        path[i:i + window_size]
        for path in paths
        if len(path) >= window_size
        for i in range(len(path) - window_size + 1)
    ]

def classify_window(window: List[str]) -> str:
    """
    Classify a window based on time t tracking type.

    Args:
        window (List[str]): List of nodes representing a time window

    Returns:
        str: Classification ('continue', 'merge', 'split', or 'unknown')
    """
    current_node = window[3] # time t node
    ttype = current_node.split("_")[-1]

    # Map node type to classification
    type_mapping = {
        "C": "continue",
        "M": "merge",
        "S": "split"
    }
    return type_mapping.get(ttype, "unknown")

def classify_windows(windows: List[List[str]]) -> Dict[str, List[List[str]]]:
    """
    Classify all windows into their respective categories.

    Args:
        windows (List[List[str]]): List of all windows to classify

    Returns:
        Dict[str, List[List[str]]]: Dictionary mapping categories to windows
    """
    classified = {"continue": [], "merge": [], "split": []}
    for window in windows:
        case_type = classify_window(window)
        if case_type in classified:
            classified[case_type].append(window)
    return classified

def list_to_dict(lst: List) -> Dict[int, List]:
    """
    Convert a list to a dictionary with integer keys.
    """
    return {i: item for i, item in enumerate(lst)}

def save_json(data: Dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    # Load graph and identify starting nodes
    print(f"Loading graph from: {GRAPH_FILE}")
    G = load_graph(GRAPH_FILE)
    sources = get_source_nodes(G)

    # Find and filter all valid paths
    all_paths = [
        path for source in sources for path in dfs_paths(G, source)
    ]
    all_paths = [
        path for path in all_paths 
        if all(is_valid_node(node) for node in path)
    ]
    all_paths = [
        path for path in all_paths if len(path) >= WINDOW_SIZE
    ]
    print(f"No. of all paths: {len(all_paths)}")

    # Generate and classify windows
    all_windows = create_sliding_windows(all_paths)
    print(f"No. of all windows: {len(all_windows)}")

    classified_windows = classify_windows(all_windows)
    for case_type, windows in classified_windows.items():
        print(f"No. of {case_type} cases: {len(windows)}")

    # Prepare and save results
    data_to_save = {
        'all_paths': list_to_dict(all_paths),
        'all_windows': list_to_dict(all_windows),
        **{
            f"{case_type}_case": list_to_dict(windows)
            for case_type, windows in classified_windows.items()
        }
    }

    os.makedirs(DFS_SAVE_FOLDER, exist_ok=True)
    for name, data in data_to_save.items():
        save_json(data, os.path.join(DFS_SAVE_FOLDER, f'{name}.json'))
    print(f"Results saved at: {DFS_SAVE_FOLDER}")

if __name__ == "__main__":
    main()