from pathlib import Path
import pandas as pd
import numpy as np
from scipy import sparse

from sklearn.svm import SVC # Dùng SVC thay cho LinearSVC để chạy kernel RBF
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.utils.utils import save_pickle


# =========================
# HELPER FOR NAIVE BAYES
# =========================
def to_dense_array(X):
    """Hàm phụ trợ: Chuyển ma trận thưa (sparse) thành ma trận đặc (dense) cho MinMaxScaler"""
    if sparse.issparse(X):
        return X.toarray()
    return X


# =========================
# LOAD DATA
# =========================
def load_features(train_path: Path, test_path: Path, label_path: Path):
    # Tự động nhận diện định dạng file (Dense từ SBERT hoặc Sparse từ TF-IDF)
    if str(train_path).endswith('.npz'):
        X_train = sparse.load_npz(train_path)
        X_test = sparse.load_npz(test_path)
    else:
        X_train = np.load(train_path)
        X_test = np.load(test_path)
    
    y_train = pd.read_csv(label_path).iloc[:, 0].values
    
    return X_train, X_test, y_train


# MODEL FACTORY
def get_model(model_type="svm", C=1.0):
    if model_type == "svm":
        # Ưu tiên kernel='rbf' thay cho tuyến tính khi làm việc với SBERT/TF-IDF mix
        return SVC(
            C=C,
            kernel='rbf',
            class_weight="balanced",
            probability=True,
            max_iter=5000
        )
    
    elif model_type == "logreg":
        return LogisticRegression(
            C=C,
            max_iter=2000,
            class_weight="balanced",
        )
    
    elif model_type == "nb":
        # Pipeline này sẽ:
        # 1. Chuyển sparse -> dense (nếu cần)
        # 2. Scale dữ liệu về khoảng [0, 1] triệt tiêu giá trị âm
        # 3. Đưa vào MultinomialNB
        return Pipeline([
            ('to_dense', FunctionTransformer(to_dense_array, accept_sparse=True)),
            ('scaler', MinMaxScaler()),
            ('clf', MultinomialNB())
        ])
    
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


# =========================
# TRAIN + VALIDATE
# =========================
def train_model(X_train, y_train, model_type="svm", C=1.0, val_size=0.2):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=42,
        stratify=y_train
    )
    
    model = get_model(model_type, C)
    model.fit(X_tr, y_tr)
    
    y_pred = model.predict(X_val)
    
    print("\n===== VALIDATION RESULTS =====")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("F1-macro:", f1_score(y_val, y_pred, average="macro"))
    print(classification_report(y_val, y_pred))
    
    return model


# =========================
# TRAIN FULL
# =========================
def train_full(X_train, y_train, model_type="svm", C=1.0):
    model = get_model(model_type, C)
    model.fit(X_train, y_train)
    return model


# =========================
# SAVE MODEL
# =========================
def save_model(model, path: Path):
    save_pickle(model, str(path))


# =========================
# MAIN PIPELINE
# =========================
def run_training(
    train_path: Path,
    test_path: Path,
    label_path: Path,
    model_save_path: Path,
    model_type: str = "svm", 
    C: float = 1.0,
    val_size: float = 0.2,
    tune_C: bool = True     
):
    print("Loading features...")
    X_train, X_test, y_train = load_features(train_path, test_path, label_path)

    print("Shape:", X_train.shape, y_train.shape)

    # =========================
    # OPTIONAL: TUNE C
    # =========================
    # Naive Bayes không có siêu tham số C nên ta tự động bỏ qua bước tuning nếu là 'nb'
    if tune_C and model_type in ["svm", "logreg"]:
        print("\nTuning C...")
        best_C = C
        best_f1 = -1

        for c in [0.1, 0.5, 1, 5, 10]:
            print(f"\n--- Testing C={c} ---")
            model = train_model(X_train, y_train, model_type, C=c, val_size=val_size)
            
            # evaluate lại nhanh
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=42, stratify=y_train)
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average="macro")

            if f1 > best_f1:
                best_f1 = f1
                best_C = c

        print(f"\nBest C found: {best_C} (F1={best_f1:.4f})")
        C = best_C
    elif model_type == "nb":
        print("\nSkipping hyperparameter tuning for Naive Bayes (No 'C' parameter).")

    # TRAIN VALIDATION MODEL
    print("\nTraining validation model...")
    _ = train_model(X_train, y_train, model_type, C, val_size)

    # TRAIN FINAL MODEL
    print("\nTraining full model...")
    final_model = train_full(X_train, y_train, model_type, C)

    # SAVE
    print("\nSaving model...")
    save_model(final_model, model_save_path)

    print("\nDONE")

    return final_model