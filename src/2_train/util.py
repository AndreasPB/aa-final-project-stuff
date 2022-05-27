import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics


def summary_report(y_test: list, y_test_pred: list, msg: str | None):
    if msg:
        print(msg)

    print(
        classification_report(
            y_test, y_test_pred, target_names=["Unhelpful", "Helpful"]
        ),
        "\nBalanced acc score: ",
        round(balanced_accuracy_score(y_test, y_test_pred), 5),
        "\nBalanced acc score(luck adjusted): ",
        round(balanced_accuracy_score(y_test, y_test_pred, adjusted=True), 5),
    )

    eval = evaluate(y_test, y_test_pred)

    print("Youden Index: ", round(eval["Youden"], 5))

    disp = metrics.ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred))
    disp.plot()
    plt.show()


def evaluate(Y, H, beta=1.0):
    tp = sum((Y == H) * (Y == 1) * 1)
    tn = sum((Y == H) * (Y == 0) * 1)
    fp = sum((Y != H) * (Y == 0) * 1)
    fn = sum((Y != H) * (Y == 1) * 1)

    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)
    youden = sensitivity - (1 - specificity)

    result = {}
    result["sensitivity"] = sensitivity
    result["specificity"] = specificity
    result["Youden"] = youden

    return result
