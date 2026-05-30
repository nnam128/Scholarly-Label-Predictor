import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, cohen_kappa_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import Optional
import warnings
import scipy.optimize as sp_opt
from functools import partial

warnings.filterwarnings("ignore")

#  Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
NUM_WORKERS = 0 if DEVICE.type == "cpu" else 2


# ── [1] Class weights helper ───────────────────────────────────────────────────
def compute_class_weights(labels: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    counts  = np.bincount(labels - 1, minlength=num_classes).astype(float)
    weights = len(labels) / (num_classes * counts)
    weights = weights / weights.mean()
    print(f"Class weights: { {i+1: round(w,3) for i,w in enumerate(weights)} }")
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


# ── [2] Optimized Rounder ──────────────────────────
class OptimizedRounder:
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        coef = np.sort(coef) 
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:   X_p[i] = 1
            elif pred < coef[1]: X_p[i] = 2
            elif pred < coef[2]: X_p[i] = 3
            elif pred < coef[3]: X_p[i] = 4
            else:                X_p[i] = 5
        return -cohen_kappa_score(y, X_p, weights='quadratic')

    def fit(self, X, y, verbose=True):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [1.5, 2.5, 3.5, 4.5]
        self.rounder = sp_opt.minimize(loss_partial, initial_coef, method='nelder-mead')
        self.coef_ = np.sort(self.rounder.x) 
        if verbose:
            print(f"Optimized Thresholds: {np.round(self.coef_, 4)}")

    def predict(self, X, coef):
        coef = np.sort(coef) 
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:   X_p[i] = 1
            elif pred < coef[1]: X_p[i] = 2
            elif pred < coef[2]: X_p[i] = 3
            elif pred < coef[3]: X_p[i] = 4
            else:                X_p[i] = 5
        return X_p.astype(int)


# ── [3] Dataset ────────────────────────────────────────────
class PaperDataset(Dataset):
    def __init__(self, texts: list, labels: Optional[list], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

def dynamic_collate_fn(batch):
    input_ids      = [item["input_ids"]      for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    input_ids      = nn.utils.rnn.pad_sequence(input_ids,      batch_first=True, padding_value=0)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    result = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "labels" in batch[0]:
        result["labels"] = torch.stack([item["labels"] for item in batch])
    return result


# ── [4] Model: BERT Regression ────────────────────────
class BERTOrdinalRegressor(nn.Module):
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased", dropout: float = 0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = out.last_hidden_state
        mask      = attention_mask.unsqueeze(-1).float()
        sum_emb   = (token_emb * mask).sum(dim=1)
        count     = mask.sum(dim=1).clamp(min=1e-9)
        pooled    = sum_emb / count
        
        return self.regressor(pooled).squeeze(-1)


# ── [5] LLRD Optimizer ─────────────────────────────────────────────────────────
def make_llrd_optimizer(model: BERTOrdinalRegressor, base_lr: float, decay: float = 0.9) -> torch.optim.AdamW:
    num_layers = model.bert.config.num_hidden_layers
    param_groups = [{"params": model.regressor.parameters(), "lr": base_lr}]
    for layer_idx in range(num_layers - 1, -1, -1):
        layer_lr = base_lr * (decay ** (num_layers - layer_idx))
        param_groups.append({
            "params": model.bert.encoder.layer[layer_idx].parameters(),
            "lr": layer_lr,
        })
    embed_lr = base_lr * (decay ** (num_layers + 1))
    param_groups.append({
        "params": model.bert.embeddings.parameters(),
        "lr": embed_lr,
    })
    return torch.optim.AdamW(param_groups, weight_decay=0.01)


# ── [6] Trainer ( Improve theo QWK và lưu Threshold riêng) ───
class BERTTrainer:
    def __init__(
        self,
        model_name:     str   = "allenai/scibert_scivocab_uncased",
        max_length:     int   = 256,
        epochs:         int   = 17,
        early_stopping: int   = 3,
        batch_size:     int   = 8,
        lr:             float = 2e-5,
        n_splits:       int   = 5,
        llrd_decay:     float = 0.9,
    ):
        self.model_name     = model_name
        self.max_length     = max_length
        self.epochs         = epochs
        self.early_stopping = early_stopping
        self.batch_size     = batch_size
        self.lr             = lr
        self.n_splits       = n_splits
        self.llrd_decay     = llrd_decay

        self.tokenizer       = AutoTokenizer.from_pretrained(model_name)
        self.models          = []
        self.fold_thresholds = []
        self.oof_final_preds = None  
        self.class_weights   = None
        self.rounder         = OptimizedRounder()

    @staticmethod
    def build_texts(df: pd.DataFrame) -> list:
        texts = []
        for _, row in df.iterrows():
            title    = str(row.get("title",    "")).strip()
            abstract = str(row.get("abstract", "")).strip()
            year     = str(row.get("year",     "")).strip()
            authors  = str(row.get("authors",  "")).strip()
            
            author_list   = [a.strip() for a in authors.split(",") if a.strip()]
            authors_short = ", ".join(author_list[:3])
            if len(author_list) > 3:
                authors_short += " et al."
                
            texts.append(
                f"year: {year} "
                f"authors: {authors_short} "
                f"title: {title} "
                #f"[SEP] abstract: {abstract}"
            )
        return texts
    
    def _train_fold(self, train_loader, val_loader, fold: int):
        model     = BERTOrdinalRegressor(self.model_name).to(DEVICE)
        optimizer = make_llrd_optimizer(model, self.lr, self.llrd_decay)

        total_steps  = len(train_loader) * self.epochs
        scheduler    = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, total_steps=total_steps, pct_start=0.1,
        )

        best_val_qwk = -1.0
        best_state   = None
        best_coef    = None
        no_improve   = 0
        
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            
            criterion = nn.SmoothL1Loss(beta=1.0, reduction='none') 

            for batch in train_loader:
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels         = batch["labels"].to(DEVICE)
                
                optimizer.zero_grad()
                preds = model(input_ids, attention_mask)
                
                raw_loss = criterion(preds, labels)
                
                label_indices = (labels.long() - 1).clamp(0, 4)
                sample_weights = self.class_weights[label_indices]
                loss = (raw_loss * sample_weights).mean()

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            val_preds, val_labels = self._predict_loader(model, val_loader)
            val_loss = np.mean((val_preds - val_labels)**2)
            
            epoch_rounder = OptimizedRounder()
            epoch_rounder.fit(val_preds, val_labels, verbose=False) 
            epoch_preds = epoch_rounder.predict(val_preds, epoch_rounder.coef_)
            val_qwk_opt = cohen_kappa_score(val_labels, epoch_preds, weights="quadratic")

            # Quyết định lưu mô hình và so sánh Improve theo val_qwk_opt
            flag = ("✔ best" if val_qwk_opt > best_val_qwk 
                    else f"(no improve {no_improve+1}/{self.early_stopping})")
            
            print(f"  Epoch {epoch+1:>2}/{self.epochs} "
                  f"train_loss={avg_loss:.4f} "
                  f"val_mse={val_loss:.4f} "
                  f"val_qwk_opt={val_qwk_opt:.4f} " 
                  f"{flag}")

            if val_qwk_opt > best_val_qwk:
                best_val_qwk = val_qwk_opt
                no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_coef  = epoch_rounder.coef_.copy() # Khóa chặt Threshold tại epoch xuất sắc này lại
            else:
                no_improve += 1
                if no_improve >= self.early_stopping:
                    print(f"  ⏹ Early stopping tại epoch {epoch+1}")
                    break

        model.load_state_dict(best_state)
        return model, best_coef # Trả về cả model và Threshold đi kèm của nó

    @staticmethod
    def _predict_loader(model, loader) -> tuple:
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                preds = model(input_ids, attention_mask)
                all_preds.extend(preds.cpu().numpy())
                if "labels" in batch:
                    all_labels.extend(batch["labels"].numpy())
        return np.array(all_preds), np.array(all_labels)

    def train(self, df: pd.DataFrame, label_col: str = "label"):
        texts  = self.build_texts(df)
        labels = df[label_col].values

        self.class_weights = compute_class_weights(labels, 5)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        self.fold_thresholds = [] 
        self.oof_final_preds = np.zeros(len(df), dtype=int) # Lưu trực tiếp nhãn int đã qua phân ngưỡng riêng

        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
            print(f"\n{'='*50}")
            print(f"Fold {fold+1}/{self.n_splits}")
            print(f"{'='*50}")

            train_texts  = [texts[i] for i in train_idx]
            val_texts    = [texts[i] for i in val_idx]
            train_labels = labels[train_idx].tolist()
            val_labels   = labels[val_idx].tolist()

            train_ds = PaperDataset(train_texts, train_labels, self.tokenizer, self.max_length)
            val_ds   = PaperDataset(val_texts,   val_labels,   self.tokenizer, self.max_length)

            train_loader = DataLoader(
                train_ds, batch_size=self.batch_size, shuffle=True,
                num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
                collate_fn=dynamic_collate_fn,
            )
            val_loader = DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
                collate_fn=dynamic_collate_fn,
            )

            # Thu thập model kèm Threshold tương ứng của fold
            model, fold_coef = self._train_fold(train_loader, val_loader, fold)
            self.models.append(model)
            self.fold_thresholds.append(fold_coef)
            print(f"  -> Fold {fold+1} Saved Threshold: {np.round(fold_coef, 4)}")

            # Tính toán và chốt nhãn OOF cho Fold này bằng chính Threshold riêng của nó
            val_preds, _ = self._predict_loader(model, val_loader)
            self.oof_final_preds[val_idx] = self.rounder.predict(val_preds, fold_coef)

        print(f"\n{'='*50}")
        # Không cần chạy Nelder-Mead toàn cục nữa, chấm điểm dựa trên nhãn kết hợp từ các Fold độc lập
        oof_qwk = cohen_kappa_score(labels, self.oof_final_preds, weights="quadratic")
        oof_f1  = f1_score(labels, self.oof_final_preds, average="macro")
        
        print(f"Overall OOF QWK (Per-Fold Thresholds) : {oof_qwk:.4f}")
        print(f"Overall OOF Macro F1                  : {oof_f1:.4f}")
        print(f"{'='*50}")
        return oof_qwk

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        texts = self.build_texts(df)
        test_ds = PaperDataset(texts, None, self.tokenizer, self.max_length)
        loader  = DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
            collate_fn=dynamic_collate_fn,
        )
        
        all_fold_preds = []
        for i, model in enumerate(self.models):
            model.eval()
            model.to(DEVICE)
            fold_preds = []
            with torch.no_grad():
                for batch in loader:
                    preds = model(batch["input_ids"].to(DEVICE),
                                  batch["attention_mask"].to(DEVICE))
                    fold_preds.extend(preds.cpu().numpy())
            
            # Khớp dự đoán thô của từng mô hình fold với chính bộ Threshold lưu trữ riêng của nó
            fold_classes = self.rounder.predict(np.array(fold_preds), self.fold_thresholds[i])
            all_fold_preds.append(fold_classes)
            
        # Ensemble bằng cách lấy Trung vị (Median) từ kết quả nhãn nguyên của 5 Fold
        return np.median(all_fold_preds, axis=0).astype(int)

    def save_models(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        for i, model in enumerate(self.models):
            path = os.path.join(save_dir, f"bert_fold_{i+1}.pt")
            torch.save(model.state_dict(), path)
            print(f"Saved: {path}")
        np.save(os.path.join(save_dir, "fold_thresholds.npy"), np.array(self.fold_thresholds))
            
    def load_models(self, save_dir: str):
        self.models = []
        for i in range(self.n_splits):
            path = os.path.join(save_dir, f"bert_fold_{i+1}.pt")
            model = BERTOrdinalRegressor(model_name=self.model_name)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.to(DEVICE)
            self.models.append(model)
            print(f"Loaded: {path}")
        coef_path = os.path.join(save_dir, "fold_thresholds.npy")
        if os.path.exists(coef_path):
            self.fold_thresholds = list(np.load(coef_path))
            print("Loaded Fold Thresholds successfully.")