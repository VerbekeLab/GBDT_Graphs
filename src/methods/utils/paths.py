import pickle
import pandas as pd
import shap
import networkx as nx
import numpy as np


def path_concatenated(path, pattern):
    i = 0
    result = []
    for els in pattern:
        if i < len(path):
            if els == path[i][0]:
                result.append(path[i])
                i = i + 1
            else:
                result.append("NA")
        else:
            result.append("NA")
    return result

def generate_paths(graph, node, path, max_length):
    paths = []
    if len(path) <= max_length:
        path.append(node)
        paths.append(path.copy())
        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                new_path = path.copy()
                paths.extend(generate_paths(graph, neighbor, new_path, max_length))
    return paths

def filter_subpaths(paths):
    """
    Filter out paths that are entirely included in another path.

    Args:
    - paths (list of lists): A list where each sublist represents a path.

    Returns:
    - list: A filtered list of paths.
    """
    # Sort the paths by length
    sorted_paths = sorted(paths, key=len)

    filtered_paths = []

    for i in range(len(sorted_paths)):
        is_subpath = False
        for j in range(i + 1, len(sorted_paths)):
            # Check if the shorter path is a prefix of a longer path
            if sorted_paths[i] == sorted_paths[j][:len(sorted_paths[i])]:
                is_subpath = True
                break
        if not is_subpath:
            filtered_paths.append(sorted_paths[i])

    return filtered_paths

def calculate_path_probabilities(tree):
    """
    Calculate the probability of following each path in the tree.

    Args:
    - tree (list of lists): A list where each sublist represents a path from root to a leaf.

    Returns:
    - dict: A dictionary where keys are paths (as tuples) and values are their probabilities.
    """
    path_probs = {}

    # For each path in the tree
    for path in tree:
        prob = 1
        for i in range(len(path) - 1):
            # Get unique children of the current node
            unique_children = set(p[i + 1] for p in tree if len(p) > i + 1 and p[i] == path[i])
            children_count = len(unique_children)
            prob *= 1 / children_count
        path_tuple = tuple(path)  # Convert list to tuple so it can be a dictionary key
        path_probs[path_tuple] = prob

    return path_probs

def generate_features(G, 
                      label_train, 
                      max_path_length, 
                      path_names_cols,
                      feature_list,
                      pattern_path
                      ):

    train_X_path_extended = None

    for node in label_train.index:
        all_paths = list(generate_paths(G, node, [], max_path_length))
        all_paths = filter_subpaths(all_paths)
        path_concatenat = [path_concatenated(path, pattern=pattern_path) for path in all_paths]
        path_concatenated_array = np.array(path_concatenat)

        feat_list = []
        for i, df in enumerate(feature_list):
            feat_list.append(df.loc[path_concatenated_array[:, i]].reset_index(drop=True))
        feat = pd.concat(feat_list, axis=1, ignore_index=True)

        feat.columns = path_names_cols
        feat['Y'] = label_train['Label'].loc[node]
        feat['group'] = node
        feat['weight'] = list(calculate_path_probabilities(all_paths).values())

        if train_X_path_extended is not None:
            train_X_path_extended = pd.concat([feat, train_X_path_extended], axis=0, ignore_index=True)
        else:
            train_X_path_extended = feat.copy()

    train_X_path_extended[path_names_cols] = train_X_path_extended[path_names_cols].apply(pd.to_numeric,
                                                                                          errors='coerce')
    Proba_check = (train_X_path_extended.groupby('group')['weight'].sum().reset_index()['weight'].round(2) == 1.0).all()
    if ~Proba_check:
        print("Error: All sum of weights of paths is not equal to one. Please double check the code")

    return train_X_path_extended