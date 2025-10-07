import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from src.methods.utils.classifiers import Nx_to_SG, train_hinsage_model, Metapath2vec
import lightgbm as lgb

def train_hinsage_full(
        graph_data_train,
        graph_data_test,
        node_type_classification_index,
        number_node_types,
        dataset, 
        head_node_type, 
        hyperparameters_hinsage
):
    SGg_train = Nx_to_SG(graph_data_train[0], graph_data_train[1], graph_data_train[2], dataset)
    SGg_test = Nx_to_SG(graph_data_test[0], graph_data_test[1], graph_data_test[2], dataset)
    y_pred_hin = train_hinsage_model(
        SGg_train,
        SGg_test, 
        graph_data_train[number_node_types+node_type_classification_index], 
        graph_data_test[number_node_types+node_type_classification_index], 
        batch_size=hyperparameters_hinsage['batch_size'], 
        num_samples=hyperparameters_hinsage['num_samples'],
        model_layer_sizes=hyperparameters_hinsage['model_layer_sizes'], 
        learning_rate=hyperparameters_hinsage['learning_rate'], 
        epochs=hyperparameters_hinsage['epochs'],
        early_stopping_patience=hyperparameters_hinsage['early_stopping_patience'], 
        verbose=1, 
        head_node_type=head_node_type
    )

    return y_pred_hin

def train_metapath2vec_full(
        graph_data_train,
        graph_data_test,
        node_type_classification_index,
        number_node_types,
        dataset,
        parameters_method,
        dimensions = 64,
        num_walks = 2,
        walk_length = 5,
        context_window_size = 3, 
        full_feature_set = False
):
    metapaths = parameters_method['metapath']
    head_node_type = parameters_method['head_node_type']

    graph_data = nx.compose(graph_data_train[0], graph_data_test[0])
    SGg = Nx_to_SG(
        graph_data, 
        pd.concat([graph_data_train[1], graph_data_test[1]]),
        pd.concat([graph_data_train[2], graph_data_test[2]]),
        dataset
        )
    # Node_targets are the node_type
    node_ids, node_embeddings, node_targets = Metapath2vec(SGg,
                                                           metapaths,
                                                           dimensions=dimensions,
                                                           num_walks=num_walks,
                                                           walk_length=walk_length,
                                                           context_window_size=context_window_size)
    
    labels = pd.concat([graph_data_train[number_node_types+node_type_classification_index], graph_data_test[number_node_types+node_type_classification_index]])

    embedding_df = pd.DataFrame(node_embeddings)
    embedding_df.index = node_ids
    target_embedding_df = embedding_df.loc[list(SGg.nodes(head_node_type))]
    embedding_fraud = target_embedding_df.merge(labels, left_index=True, right_index=True)
    embedding_fraud.sort_index(inplace=True)
    embedding_fraud.columns = ["Meta_"+str(i) for i in range(dimensions)] + list(labels.columns)

    df_train = graph_data_train[1].copy().drop('NA')
    df_test = graph_data_test[1].copy().drop('NA')

    X = embedding_fraud.iloc[:, :-1]
    y = embedding_fraud.iloc[:, -1]

    if full_feature_set:
        original_features = pd.concat([df_train, df_test]).loc[X.index]
        X = pd.concat([X, original_features], axis=1)

    X_train = X.loc[df_train.index]
    y_train = y.loc[df_train.index]
    X_test = X.loc[df_test.index]
    y_test = y.loc[df_test.index]

    print("Building the model...")

    # Prepare datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)

    # Set parameters
    params_model = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }

    # Train the model
    num_round = 500
    bst = lgb.train(params_model, train_data, num_round, valid_sets=[test_data], callbacks=[lgb.early_stopping(10)])

    # Predict probabilities using the model
    y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)

    return y_pred, bst
