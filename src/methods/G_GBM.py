# Proprietary
from src.utils.param_dict import path_names_cols_construct, feature_list_construct
from src.methods.utils.classifiers import train_lgb_model
from src.methods.utils.paths import generate_features

def G_GBM(graph_data_train, graph_data_test, node_type_classification_index, number_node_types, parameters_data, parameters_method):
    path_names_cols = path_names_cols_construct(graph_data_train, parameters_method['pattern_path'], parameters_method['neighbourhood'], parameters_data['node_types_dict'])


    params_paths_train = {
        'max_path_length':parameters_method['max_path_length'],
        'path_names_cols': path_names_cols,
        'feature_list': feature_list_construct(graph_data_train, parameters_method['pattern_path'], parameters_data['node_types_dict']),
        'pattern_path': parameters_method['pattern_path']
    }

    params_paths_test = {
        'max_path_length':parameters_method['max_path_length'],
        'path_names_cols': path_names_cols,
        'feature_list': feature_list_construct(graph_data_test, parameters_method['pattern_path'], parameters_data['node_types_dict']),
        'pattern_path': parameters_method['pattern_path']
    }

    train_X_path_extended, test_X_path_extended = generate_features(
        graph_data_train[0], 
        graph_data_train[number_node_types+node_type_classification_index], # Label of that node type
        **params_paths_train
        ), \
                                                generate_features(
                                                    graph_data_test[0],
                                                    graph_data_test[number_node_types+node_type_classification_index], # Label of that node type
                                                    **params_paths_test
                                                    )

    preds, bst = train_lgb_model(train_X_path_extended, test_X_path_extended, path_names_cols)
    return preds, bst, test_X_path_extended, path_names_cols