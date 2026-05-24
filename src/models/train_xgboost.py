import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np


class XGBoostTrainer:
    def __init__(self, params=None):
        self.params = params if params else {
            'objective': 'multi:softprob',
            'num_class': 5,
            'eval_metric': 'mlogloss',
            'verbosity': 0,
            'booster': 'gbtree',
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'seed': 42,
        }
        self.models = []

    def train(self, X, y, n_splits=5):
        """
        Train với Cross Validation để tránh Overfitting
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        X = np.asarray(X)
        y = np.asarray(y)
        oof_preds = np.zeros((X.shape[0], self.params['num_class']))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            model = xgb.train(
                self.params,
                dtrain,
                evals=[(dval, 'valid')],
                num_boost_round=500,
                early_stopping_rounds=50,
                verbose_eval=False,
            )

            self.models.append(model)
            oof_preds[val_idx] = model.predict(dval).reshape(-1, self.params['num_class'])

        # Tính Macro F1 score trên toàn bộ tập OOF (Out-of-fold)
        final_preds = np.argmax(oof_preds, axis=1)
        macro_f1 = f1_score(y, final_preds, average='macro')
        print(f"\nOverall Macro F1: {macro_f1:.4f}")
        return macro_f1

    def predict(self, X_test):
        # Lấy trung bình dự đoán từ tất cả các fold
        X_test = np.asarray(X_test)
        dtest = xgb.DMatrix(X_test)
        test_preds = np.zeros((X_test.shape[0], self.params['num_class']))
        for model in self.models:
            test_preds += model.predict(dtest).reshape(-1, self.params['num_class'])
        return np.argmax(test_preds / len(self.models), axis=1)