import os
import pathlib
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
def load_data():
    """Load data from the specified data source.

    Args:
        data_source (str): The path or identifier of the data source.

    Returns:
        edges: Loaded edges data.
        nodes: Loaded nodes data.
    """
    if os.path.exists(DATA_DIR):
        edges = pd.read_csv(pathlib.Path(DATA_DIR) / "darkweb-edges.ss", sep=";")
        nodes = pd.read_csv(pathlib.Path(DATA_DIR) / "darkweb-nodes.ss", sep=";")
    
    return nodes, edges

def transform_node_ids(nodes: pd.DataFrame) -> pd.DataFrame:
    """Transform node IDs by replacing the original Ids with new sequential integers and storing the original Ids in a new column 'Name'.

    Args:
        nodes (pd.DataFrame): input nodes df

    Returns:
        pd.DataFrame: transformed nodes df with Id as indexes and name as a column
    """
    nodes["Name"] = nodes["Id"]
    nodes["Id"] = range(len(nodes))
    return nodes

def transform_edge_ids(edges: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Transform edge source and target names to their corresponding node Ids.

    Args:
        edges (pd.DataFrame): _input edges df
        nodes (pd.DataFrame): nodes df transformed with transform_node_ids

    Returns:
        pd.DataFrame: transformed edges df with Source and Target as node Ids
    """
    name_to_id = dict(zip(nodes["Name"], nodes["Id"]))
    edges = edges.copy()
    
    edges["Source"] = edges["Source"].map(name_to_id)
    edges["Target"] = edges["Target"].map(name_to_id)

    missing_sources = edges["Source"].isna().sum()
    missing_targets = edges["Target"].isna().sum()
    if missing_sources or missing_targets:
        print(f"Missing Source IDs: {missing_sources}, Missing Target IDs: {missing_targets}")

    edges = edges.dropna(subset=["Source", "Target"]).astype({"Source": int, "Target": int})

    return edges






