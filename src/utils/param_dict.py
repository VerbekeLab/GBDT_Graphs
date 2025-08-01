import numpy as np


def path_names_cols_construct(graph_data, path, neighbourhood, node_types_dict):
    path_names_cols_list = []
    for i in range(len(path)):
        node_type = path[i]
        node_type_index = node_types_dict[node_type]
        path_names_cols_list.append(neighbourhood[i] + ' ' + graph_data[node_type_index].columns.values)
    
    return np.concatenate(path_names_cols_list)

def feature_list_construct(graph_data, pattern_path, node_types_dict):
    feature_list = []
    for i in range(len(pattern_path)):
        feature_list.append(
            graph_data[node_types_dict[pattern_path[i]]]
        )
    return feature_list

