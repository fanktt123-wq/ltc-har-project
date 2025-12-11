from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def compute_metrics(y_true, y_pred, num_classes=6):
    """计算准确率、精确率、召回率、F1分数"""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    return {
        'accuracy': acc,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    }

