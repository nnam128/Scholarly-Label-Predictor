import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np

class LightGBMTrainer:
    def __init__(self, params=None):
        self.params = params if params else {
            'objective': 'multiclass',
            'num_class': 5,
            'metric': 'multi_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1
        }
        self.models = []

    def train(self, X, y, n_splits=5):
        """
        Train với Cross Validation để tránh Overfitting
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_preds = np.zeros((X.shape[0], self.params['num_class']))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X = np.asarray(X)
            y = np.asarray(y)
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            model = lgb.train(
                self.params,
                dtrain,
                valid_sets=[dval],
                valid_names=['valid'],
                num_boost_round=500,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                ]
            )
            
            self.models.append(model)
            oof_preds[val_idx] = model.predict(X_val)
            
        # Tính Macro F1 score trên toàn bộ tập OOF (Out-of-fold)
        final_preds = np.argmax(oof_preds, axis=1)
        macro_f1 = f1_score(y, final_preds, average='macro')
        print(f"\nOverall Macro F1: {macro_f1:.4f}")
        return macro_f1

    def predict(self, X_test):
        # Lấy trung bình dự đoán từ tất cả các fold
        test_preds = np.zeros((X_test.shape[0], self.params['num_class']))
        for model in self.models:
            test_preds += model.predict(X_test)
        return np.argmax(test_preds / len(self.models), axis=1)