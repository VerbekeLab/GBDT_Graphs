from src.methods.utils.classifiers import Nx_to_SG, train_hinsage_model

def train_hinsage_full(
        graph_data_train,
        graph_data_test,
        node_type_classification_index,
        number_node_types,
        dataset, 
        head_node_type
):
    SGg_train = Nx_to_SG(graph_data_train[0], graph_data_train[1], graph_data_train[2], dataset)
    SGg_test = Nx_to_SG(graph_data_test[0], graph_data_test[1], graph_data_test[2], dataset)
    y_pred_hin = train_hinsage_model(
        SGg_train,
        SGg_test, 
        graph_data_train[number_node_types+node_type_classification_index], 
        graph_data_test[number_node_types+node_type_classification_index], 
        batch_size=1000, 
        num_samples=[10, 5],
        model_layer_sizes=[32, 32], 
        learning_rate=0.01, 
        epochs=10,
        early_stopping_patience=5, 
        verbose=1, 
        head_node_type=head_node_type
    )

    return y_pred_hin