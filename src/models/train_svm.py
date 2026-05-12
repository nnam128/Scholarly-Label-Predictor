from pathlib import Path
import pandas as pd
from scipy import sparse

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.utils.utils import save_pickle


# =========================
# LOAD DATA
# =========================
def load_features(train_path: Path, test_path: Path, label_path: Path):
    X_train = sparse.load_npz(train_path)
    X_test = sparse.load_npz(test_path)
    
    y_train = pd.read_csv(label_path).iloc[:, 0].values
    
    return X_train, X_test, y_train


# =========================
# MODEL FACTORY
# =========================
def get_model(model_type="svm", C=1.0):
    if model_type == "svm":
        return LinearSVC(
            C=C,
            class_weight="balanced",
            max_iter=5000
        )
    
    elif model_type == "logreg":
        return LogisticRegression(
            C=C,
            max_iter=2000,
            class_weight="balanced",
        )
    
    elif model_type == "nb":
        return MultinomialNB()
    
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
    model_type: str = "svm",   # 🔥 NEW
    C: float = 1.0,
    val_size: float = 0.2,
    tune_C: bool = True       # 🔥 NEW
):
    print("✔ Loading features...")
    X_train, X_test, y_train = load_features(train_path, test_path, label_path)

    print("Shape:", X_train.shape, y_train.shape)

    # =========================
    # OPTIONAL: TUNE C
    # =========================
    if tune_C and model_type in ["svm", "logreg"]:
        print("\n✔ Tuning C...")
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

        print(f"\n✔ Best C found: {best_C} (F1={best_f1:.4f})")
        C = best_C

    # =========================
    # TRAIN VALIDATION MODEL
    # =========================
    print("\n✔ Training validation model...")
    _ = train_model(X_train, y_train, model_type, C, val_size)

    # =========================
    # TRAIN FINAL MODEL
    # =========================
    print("\n✔ Training full model...")
    final_model = train_full(X_train, y_train, model_type, C)

    # =========================
    # SAVE
    # =========================
    print("\n✔ Saving model...")
    save_model(final_model, model_save_path)

    print("\n✔ DONE")

    return final_model