import lightgbm as lgb

def train_lgb_model(train_X_path_extended, test_X_path_extended, path_names_cols):
    # Prepare datasets
    train_data = lgb.Dataset(train_X_path_extended[path_names_cols],
                             label=train_X_path_extended['Y'], weight=train_X_path_extended['weight'])

    test_data = lgb.Dataset(test_X_path_extended[path_names_cols],
                             label=test_X_path_extended['Y'], weight=test_X_path_extended['weight'])

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
    y_pred = bst.predict(test_X_path_extended[path_names_cols], num_iteration=bst.best_iteration)

    preds = test_X_path_extended[['group','Y','weight']].copy()
    preds['Y_hat'] = y_pred
    preds['Y_hat_weighted'] = y_pred * preds['weight']

    return preds, bst


def train_lgb_model_vanilla(comp_train, comp_test, labs_train, labs_test):
    # Prepare datasets
    train_data = lgb.Dataset(comp_train.loc[labs_train.index], label=labs_train)

    test_data = lgb.Dataset(comp_test.loc[labs_test.index], label=labs_test)

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
    y_pred = bst.predict(comp_test.loc[labs_test.index], num_iteration=bst.best_iteration)
    return y_pred, bst

import numpy as np
import pandas as pd
from sklearn import preprocessing
import stellargraph as sg
from tensorflow.keras import layers, optimizers, losses, Model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def train_hinsage_model(G_train, G_test, label_train, label_test, batch_size=1000, num_samples=[10, 5],
                        model_layer_sizes=[32, 32], learning_rate=0.005, epochs=10,
                        early_stopping_patience=5, verbose=1, head_node_type="company"):
    # Encoding target labels
    target_encoding = preprocessing.LabelBinarizer()
    train_targets = target_encoding.fit_transform(label_train)
    val_targets = target_encoding.transform(label_test)

    # Setting up the generator
    generator_train = sg.mapper.HinSAGENodeGenerator(
        G_train, batch_size=batch_size, num_samples=num_samples, head_node_type=head_node_type
    )
    generator_test = sg.mapper.HinSAGENodeGenerator(
        G_test, batch_size=batch_size, num_samples=num_samples, head_node_type=head_node_type
    )
    train_gen = generator_train.flow(label_train.index, label_train)
    val_gen = generator_test.flow(label_test.index, label_test)

    # Creating the model
    base_model = sg.layer.HinSAGE(model_layer_sizes, generator=generator_train, bias=True, dropout=0.5)
    x_in, x_out = base_model.in_out_tensors()
    prediction = layers.Dense(1, activation="sigmoid")(x_out)
    model = Model(inputs=x_in, outputs=prediction)
    model.compile(loss=losses.binary_crossentropy, optimizer=optimizers.Adam(lr=learning_rate), metrics=['AUC'])

    cbs = [EarlyStopping(monitor="val_loss", mode="min", patience=early_stopping_patience)]

    # Train the model
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=verbose, shuffle=False,
                        callbacks=cbs)

    # Plot training history
    def plot_loss(history, label, n, loss):
        plt.semilogy(history.epoch, history.history[loss], color=colors[n], label='Train ' + label)
        plt.semilogy(history.epoch, history.history['val_' + loss], color=colors[n], label='Val ' + label,
                     linestyle="--")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plot_loss(history, "Mean.Agg. ", 0, "auc")
    plt.legend()
    plt.show()

    # Making predictions
    all_mapper = generator_test.flow(label_test.index, label_test) #this
    # Without label_test, it gave an error: ValueError: Data is expected to be in format `x`, `(x,)`, `(x, y)`, or `(x, y, sample_weight)`, found: (<tf.Tensor 'IteratorGetNext:0' shape=(None, None, None) dtype=float32>, <tf.Tensor 'IteratorGetNext:1' shape=(None, None, None) dtype=float32>, <tf.Tensor 'IteratorGetNext:2' shape=(None, None, None) dtype=float32>, <tf.Tensor 'IteratorGetNext:3' shape=(None, None, None) dtype=float32>, <tf.Tensor 'IteratorGetNext:4' shape=(None, None, None) dtype=float32>, <tf.Tensor 'IteratorGetNext:5' shape=(None, None, None) dtype=float32>)
    all_predictions = model.predict(all_mapper)

    return pd.Series(all_predictions.flatten())

from stellargraph import StellarGraph
def Nx_to_SG(G,type1,type2, dataset):
    if dataset in ["insurance_a", "insurance_c", "insurance_gb"]:
        return Nx_to_SG_ins(G,type1,type2)
    elif dataset in ["hcp", "hcp_gb"]:
        return Nx_to_SG_hcp(G,type1,type2)
    else:
        raise ValueError(f"No implementation for dataset: {dataset}")

def Nx_to_SG_ins(G,comp,admin):
    def get_limited_dummies(df, columns, max_categories=10):
        df_copy = df.copy()
        for col in columns:
            # Find the (max_categories-1) most frequent categories
            top_categories = df_copy[col].value_counts().head(max_categories - 1).index.tolist()

            # Where the category is not in top_categories, set it as 'Other'
            df_copy[col] = df_copy[col].where(df_copy[col].isin(top_categories), 'Other')

        # Apply pd.get_dummies
        result = pd.get_dummies(df_copy, columns=columns, drop_first=True)
        return result
    sgG = StellarGraph.from_networkx(G, node_features={"company":
        get_limited_dummies(comp, columns=['Legal Form', 'Localisation district category'], max_categories=15).drop('NA'),
                            "administrator": admin.drop('NA')}) # "fraud":fraud
    return sgG

def Nx_to_SG_hcp(G,prov,ben):
    sgG = StellarGraph.from_networkx(
        G, 
        node_features={
            "provider": prov.drop('NA'),
            "beneficiary": ben.drop('NA')
        }
    )

    return sgG