""" utils.py """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score

def optimal_cutoff(target: np.array, predicted: np.array) -> float:
    """
    Find the optimal probability cutoff point for classification.
    -------------------------------------------------------------
    target: True labels
    predicted: positive probability predicted by the model
        i.e. model.predict_proba(X_test)[:, 1], NOT 0/1 prediction array
    Returns
    --------
    cut-off value
    """
    
    fpr, tpr, threshold = roc_auc_score(target, predicted)
    threshold_indices: int = np.arange(len(tpr))
    roc: pd.DataFrame = pd.DataFrame({
        "tf": pd.Series(tpr - (1-fpr), index=threshold_indices),
        "threshold": pd.Series(threshold, index=threshold_indices)
    })
    roc_threshold = roc.iloc[(roc["tf"] - 0).abs().argsort()[:1]]
    
    return round(list(roc_threshold["threshold"])[0], 2)


def plot_confusion_matrix(y_true: np.array, y_pred: np.array) -> None:
    # Confusion matrix
    matrix = confusion_matrix(y_true, y_pred)
    data = matrix.transpose()
    
    _, ax = plt.subplot()
    ax.matshow(data, cmap="Blues")
    
    # Printing exact numbers
    for (i, j), z in np.ndenumerate(data):
        ax.text(j, i, f'{z}', ha="center", va="center")
    
    # axis formatting
    plt.xticks([])
    plt.yticks([])
    plt.title("True label\n 0 {} 1\n".format(" "*18), fontsize=14)
    plt.ylabel("True label\n 0 {} 1\n".format(" "*18), fontsize=14)
    plt.show()

def draw_roc_curve(y_true: np.array, y_proba: np.array) -> None:
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    _, ax = plt.subplots()
    ax.plot(fpr, tpr, color="r")
    ax.plot([0, 1], [0, 1], color="y", linestyle="--")
    ax.fill_between(fpr, tpr, label=f"AUC: {round(roc_auc_score(y_true, y_proba), 3)}")
    ax.set_aspect(0.9)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.legend()
    plt.show()

def summarize_results(y_true: np.array, y_pred: np.array) -> None:
    """
    Provides performance metrics
    """
    
    print("\n=========================")
    print("        RESULTS")
    print("=========================")

    print("Accuracy: ", accuracy_score(y_true, y_pred).round(2))
    conf_matrix = confusion_matrix(y_true, y_pred)
    sensitivity = round(conf_matrix[1, 1]/(conf_matrix[1, 1] + conf_matrix[1, 0]), 2)
    specificity = round(conf_matrix[0, 0]/(conf_matrix[0, 0] + conf_matrix[0, 1]), 2)
    
    ppv = round(conf_matrix[1, 1]/(conf_matrix[1, 1] + conf_matrix[0, 1]), 2)
    npv = round(conf_matrix[0, 0]/(conf_matrix[0, 0] + conf_matrix[1, 0]), 2)
    
    print("-------------------------")
    print("sensitivity: ", sensitivity)
    print("specificity: ", specificity)
    
    print("-------------------------")
    
    print("positive predictive value: ", ppv)
    print("negative predictive value: ", npv)
    
    print("-------------------------")
    print("precision: ", precision_score(y_true, y_pred).round(2))
    print("recall: ", recall_score(y_true, y_pred).round(2))