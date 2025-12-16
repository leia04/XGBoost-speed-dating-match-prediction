import numpy as np

def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return float(np.mean(y_true == y_pred))

def print_accuracy(y_train, y_pred_train, y_test, y_pred_test) -> None:
    train_acc = accuracy(y_train, y_pred_train)
    test_acc = accuracy(y_test, y_pred_test)
    print(f"Train Accuracy: {train_acc}")
    print(f"Test Accuracy: {test_acc}")
