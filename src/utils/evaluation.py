import numpy as np
import matplotlib.pyplot as plt
import shap
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def plot_evaluation_curves(y_test, y_preds, model_names, factor_reduce = 10, dataset=''):
    plt.figure(figsize=(18, 5))

    # 1. ROC curve
    plt.subplot(1, 3, 1)
    for y_pred, name in zip(y_preds, model_names):
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        if fpr.__len__() > 100:
            plt.plot(fpr[::factor_reduce], tpr[::factor_reduce], lw=2, label='%s (area = %0.2f)' % (name, roc_auc))
        else:
            plt.plot(fpr, tpr, lw=2, label='%s (area = %0.2f)' % (name, roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")

    # 2. Precision-recall curve
    plt.subplot(1, 3, 2)
    for y_pred, name in zip(y_preds, model_names):
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        average_precision = average_precision_score(y_test, y_pred)
        if recall.__len__() > 100:
            plt.step(recall[::factor_reduce], precision[::factor_reduce], where='post', label='%s AP=%0.2f' % (name, average_precision))
        else:
            plt.step(recall, precision, where='post', label='%s AP=%0.2f' % (name, average_precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('PrecisionRecall curve')
    plt.legend(loc="upper right")

    # 3. Lift chart
    plt.subplot(1, 3, 3)
    for y_pred, name in zip(y_preds, model_names):
        sorted_indices = np.argsort(y_pred)[::-1]
        sorted_y_test = y_test[sorted_indices]
        cumulative_gain = np.cumsum(sorted_y_test) / np.sum(sorted_y_test)
        lift_x = np.arange(len(y_test)) / len(y_test)
        if lift_x.__len__() > 100:
            plt.plot(lift_x[::factor_reduce], cumulative_gain[::factor_reduce], label='%s' % name)
        else:
            plt.plot(lift_x, cumulative_gain, label='%s' % name)
    plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
    plt.xlabel('Percentage of samples')
    plt.ylabel('Cumulative gain')
    plt.title('Lift Chart')
    plt.legend(loc="upper left")
    print("Performance measurement for test set of " + str(y_test.shape[0]) + " Samples, and " + str(
        y_test.sum()) + " fraud labels. Class balance = " + str(round(y_test.sum() / y_test.shape[0], 2)))
    
    plt.subplots_adjust(top=0.85)

    plt.tight_layout()

    plt.savefig(f"res/plot_results_{dataset}.pdf")
    plt.close()

def plot_shap_importance(bst, test_X_path_extended, path_names_cols, dataset=''):
    # SHAP explanation
    explainer = shap.Explainer(bst)
    shap_values = explainer(test_X_path_extended[path_names_cols])

    # Convert to DataFrame and get absolute values
    shaps_paths = pd.DataFrame(shap_values[:, :, 1].values).abs()
    shaps_paths.columns = path_names_cols

    # Merge with predictions dataframe
    preds = pd.concat([test_X_path_extended[['group', 'Y', 'weight']], shaps_paths], axis=1)

    # Apply weights
    preds[path_names_cols] = preds[path_names_cols].multiply(preds["weight"], axis="index")

    # Sum the SHAP values
    sums = preds.groupby('group')[path_names_cols].mean().sum()

    color_map = {
        'H(a)': 'red',
        'H(c)': 'orange',
        'N1(a': 'blue',
        'N1(c': 'green',
        'N2(a': 'yellow',
        'N2(c': 'purple'
    }
    # Map the list to colors
    colors = [color_map[item] for item in [paths[0:4] for paths in path_names_cols]]

    # Plotting
    sums[::-1].plot(kind='barh', color=colors)
    plt.title('Variable importance aggregated (over the test set)')
    plt.ylabel('Variable N_i in neighborhood of distance i')
    plt.xlabel('Sum of (weighted, grouped) absolute SHAP values over head nodes')
    plt.tight_layout()
    plt.savefig(f"res/plot_shap_importance_{dataset}.pdf")