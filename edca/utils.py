from sklearn import metrics

def evo_mcc_metric(y_test, y_pred, y_proba=None):
    """MCC metric for the evolutionary algorithm"""
    m = metrics.matthews_corrcoef(y_test, y_pred)
    return 1 - m


def evo_f1_metric(y_test, y_pred, y_proba=None):
    """F1 metric for the evolutionary algorithm"""
    m = metrics.f1_score(y_test, y_pred, average='macro')
    return 1 - m

def evo_precision_metric(y_test, y_pred, y_proba=None):
    """F1 metric for the evolutionary algorithm"""
    m = metrics.precision_score(y_test, y_pred, average='macro')
    return 1 - m

def evo_recall_metric(y_test, y_pred, y_proba=None):
    """F1 metric for the evolutionary algorithm"""
    m = metrics.recall_score(y_test, y_pred, average='macro')
    return 1 - m

def evo_roc_auc_metric(y_test, y_pred, y_proba=None):
    m = metrics.roc_auc_score(y_test, y_proba, average='macro')
    return 1 - m

def evo_accuracy_metric(y_test, y_pred, y_proba=None):
    """F1 metric for the evolutionary algorithm"""
    m = metrics.accuracy_score(y_test, y_pred)
    return 1 - m


def evo_metric(metric_name):
    """Method to select the metric to use on the evolutionary algorithm based on metric's name"""
    if metric_name == 'mcc':
        return evo_mcc_metric
    elif metric_name == 'f1':
        return evo_f1_metric
    elif metric_name == 'accuracy':
        return evo_accuracy_metric
    elif metric_name == 'precision':
        return evo_precision_metric
    elif metric_name == 'recall':
        return evo_recall_metric
    elif metric_name == 'roc_auc':
        return evo_roc_auc_metric
    else:
        raise ValueError('No metric with that Name')