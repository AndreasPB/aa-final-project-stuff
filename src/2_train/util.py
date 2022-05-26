import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

def summary_report(y_test, y_test_pred, msg):
    print(msg)
    print(classification_report(y_test, y_test_pred, target_names=["Unhelpful", "Helpful"]))
    print("Balanced acc score: ", balanced_accuracy_score(y_test, y_test_pred, adjusted=True))
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred))
    disp.plot()
    plt.show()