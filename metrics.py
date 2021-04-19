from sklearn.metrics import accuracy_score, roc_auc_score


class Accuracy:
    @staticmethod
    def score(y_true, y_pred):
        return accuracy_score(y_true, (y_pred >= 0.5).astype(int))


class AUC:
    @staticmethod
    def score(y_true, y_pred):
        return roc_auc_score(y_true, y_pred)
