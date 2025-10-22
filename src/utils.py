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

