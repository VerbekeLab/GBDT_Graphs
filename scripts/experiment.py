# LOAD MODULES
# Standard library
import os
import sys
import itertools
import warnings

# NOTE: Your script is not in the root directory. We must hence change the system path
DIR = os.path.dirname(os.path.abspath(__file__)) 
os.chdir(DIR+"/../")
sys.path.append(DIR+"/../")

# Proprietary
from src.data.graph_data import load_data
from src.utils.setup import load_config
from src.methods.G_GBM import *
from src.methods.utils.classifiers import train_lgb_model_vanilla
from src.methods.network import train_hinsage_full
from src.utils.evaluation import plot_evaluation_curves, plot_shap_importance

dataset = load_config("config/data/config.yaml")['parameters']['dataset']
parameters_data = load_config("config/data/config.yaml")[dataset]
parameters_method = load_config("config/methods/config.yaml")[dataset]
number_node_types = parameters_data['number_node_types']
head_node_type = parameters_method['head_node_type']

graph_data_train = load_data(dataset, parameters_data['data_path_train'], number_node_types=number_node_types)
graph_data_test = load_data(dataset, parameters_data['data_path_test'], number_node_types=number_node_types)

node_type_classification = parameters_method['pattern_path'][0]
node_type_classification_index = parameters_data['node_types_dict'][node_type_classification]

models_to_train = [
    'G-GBM', 
    'LGB', 
    'HINSage'
    ]

# G-GBM
if 'G-GBM' in models_to_train:
    print("Training G-GBM model...")
    preds, bst, test_X_path_extended, path_names_cols = G_GBM(
        graph_data_train=graph_data_train,
        graph_data_test=graph_data_test,
        node_type_classification_index=node_type_classification_index,
        number_node_types=number_node_types,
        parameters_data=parameters_data,
        parameters_method=parameters_method
    )

# Baseline lgb
if 'LGB' in models_to_train:
    print("Training Baseline LGB model...")
    preds_vanilla, bst2 = train_lgb_model_vanilla(
        comp_train=graph_data_train[node_type_classification_index], 
        comp_test=graph_data_test[node_type_classification_index], 
        labs_train=graph_data_train[number_node_types+node_type_classification_index], 
        labs_test=graph_data_test[number_node_types+node_type_classification_index]
    )

# HINSage
if 'HINSage' in models_to_train:
    print("Training HINSage model...")
    y_pred_hin = train_hinsage_full(
        graph_data_train=graph_data_train,
        graph_data_test=graph_data_test,
        node_type_classification_index=node_type_classification_index,
        number_node_types=number_node_types,
        dataset=dataset, 
        head_node_type=head_node_type
    )

# Evaluation 
plot_evaluation_curves(preds.groupby('group')['Y'].mean().loc[graph_data_test[number_node_types+node_type_classification_index].index],
                       y_preds = [preds.groupby('group')['Y_hat_weighted'].sum().loc[graph_data_test[number_node_types+node_type_classification_index].index],
                                          preds_vanilla, y_pred_hin],
                       model_names = models_to_train, factor_reduce=20, dataset=dataset)

# Explanations
plot_shap_importance(bst, test_X_path_extended, path_names_cols, dataset=dataset)