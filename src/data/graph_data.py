# LOAD MODULES
# Standard library
from typing import Union, Optional, Dict, Any

# Proprietary

# Third party
import pickle

# DATA LOADING FUNCTION
def load_data(
    dataset: str,
    path: str, 
    number_node_types: int = 1
):
    
    if dataset == "insurance":
        return load_data_insurance(path, number_node_types=number_node_types)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def load_data_insurance(
    data_path: str, 
    number_node_types: int = 1
):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    for i in range(1, number_node_types + 1):
        num_features = data[i].shape[1]
        data[i].loc['NA'] = tuple([None] * num_features)
        data[i] = data[i].astype(float)

    return data

class NetworkData:
    def __init__(self, data, number_node_types: int = 1, ):
        self.data = data
        self.number_node_types = number_node_types

    def get_graph(self):
        return self.data[0]

    def get_data(self, node_type: int = 1):
        if node_type > self.number_node_types:
            raise ValueError(f"Node type {node_type} exceeds number of node types {self.number_node_types}.")
        return self.data[node_type]
    
    def get_label(self, node_type: int = 1):
        if node_type > self.number_node_types*2:
            raise ValueError(f"Node type {node_type} exceeds number of node types {self.number_node_types}.")
        return self.data[node_type].get('label', None)