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

# ── [0] Device & Settings ──────────────────────────────────────────────────────
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


# ── [2] Optimized Rounder ──────────────────────────────────────────────────────
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


# ── [3] Dataset ────────────────────────────────────────────────────────────────
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


# ── [4] Model: BERT Regression (Đã nâng cấp) ──────────────────────────────────
class BERTOrdinalRegressor(nn.Module):
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        
        # [CẢI TIẾN]: Multi-Sample Dropout giúp model ổn định hơn
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)])
        
        # Kích thước input là hidden * 2 vì ta sẽ nối CLS pooling và Mean pooling
        self.fc = nn.Linear(hidden * 2, 1)
        
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = out.last_hidden_state
        
        # [CẢI TIẾN]: CLS Pooling
        cls_token = token_emb[:, 0, :]
        
        # [CẢI TIẾN]: Mean Pooling
        mask      = attention_mask.unsqueeze(-1).float()
        sum_emb   = (token_emb * mask).sum(dim=1)
        count     = mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = sum_emb / count
        
        # [CẢI TIẾN]: Concatenate CLS và Mean Pooling để biểu diễn phong phú hơn
        concat_pooled = torch.cat((cls_token, mean_pooled), dim=1)
        
        # [CẢI TIẾN]: Đi qua Multi-Sample Dropout và lấy trung bình
        outputs = torch.mean(
            torch.stack([self.fc(dropout(concat_pooled)) for dropout in self.dropouts], dim=0), 
            dim=0
        )
        
        return outputs.squeeze(-1)


# ── [5] LLRD Optimizer (Đã loại bỏ Decay cho Bias/LayerNorm) ─────────────────
def make_llrd_optimizer(model: BERTOrdinalRegressor, base_lr: float, decay: float = 0.9) -> torch.optim.AdamW:
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = []
    num_layers = model.bert.config.num_hidden_layers

    # Nhóm tham số của tầng Linear (Head)
    param_groups.append({
        "params": [p for n, p in model.fc.named_parameters() if not any(nd in n for nd in no_decay)],
        "lr": base_lr, "weight_decay": 0.01
    })
    param_groups.append({
        "params": [p for n, p in model.fc.named_parameters() if any(nd in n for nd in no_decay)],
        "lr": base_lr, "weight_decay": 0.0
    })

    # Nhóm tham số của các tầng Encoder (giảm LR dần)
    for layer_idx in range(num_layers - 1, -1, -1):
        layer_lr = base_lr * (decay ** (num_layers - layer_idx))
        layer = model.bert.encoder.layer[layer_idx]
        
        param_groups.append({
            "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
            "lr": layer_lr, "weight_decay": 0.01
        })
        param_groups.append({
            "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
            "lr": layer_lr, "weight_decay": 0.0
        })

    # Nhóm tham số của Embeddings
    embed_lr = base_lr * (decay ** (num_layers + 1))
    param_groups.append({
        "params": [p for n, p in model.bert.embeddings.named_parameters() if not any(nd in n for nd in no_decay)],
        "lr": embed_lr, "weight_decay": 0.01
    })
    param_groups.append({
        "params": [p for n, p in model.bert.embeddings.named_parameters() if any(nd in n for nd in no_decay)],
        "lr": embed_lr, "weight_decay": 0.0
    })

    return torch.optim.AdamW(param_groups)


# ── [6] Trainer (Tối ưu Training Loop & Logic Early Stopping) ──────────────────
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

        self.tokenizer     = AutoTokenizer.from_pretrained(model_name)
        self.models        = []
        self.oof_raw_preds = None  
        self.class_weights = None
        self.rounder       = OptimizedRounder()

    def build_texts(self, df: pd.DataFrame) -> list:
        texts = []
        # Dùng token [SEP] của tokenizer để phân tách rõ ràng mạch ngữ nghĩa
        sep = self.tokenizer.sep_token if self.tokenizer.sep_token else "[SEP]"
        
        for _, row in df.iterrows():
            title    = str(row.get("title",    "")).strip()
            abstract = str(row.get("abstract", "")).strip()
            year     = str(row.get("year",     "")).strip()
            authors  = str(row.get("authors",  "")).strip()
            
            author_list   = [a.strip() for a in authors.split(",") if a.strip()]
            authors_short = ", ".join(author_list[:3])
            if len(author_list) > 3:
                authors_short += " et al."
                
            # [CẢI TIẾN]: Đẩy title và abstract lên trước để BERT chú ý tốt hơn, metadata ở cuối
            texts.append(
                f"{title} {sep} {abstract} {sep} year: {year}, authors: {authors_short}"
            )
        return texts
    
    def _train_fold(self, train_loader, val_loader, fold: int):
        model     = BERTOrdinalRegressor(self.model_name).to(DEVICE)
        optimizer = make_llrd_optimizer(model, self.lr, self.llrd_decay)

        total_steps  = len(train_loader) * self.epochs
        scheduler    = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, total_steps=total_steps, pct_start=0.1,
        )

        # [CẢI TIẾN]: Khởi tạo GradScaler cho AMP (Automatic Mixed Precision)
        scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

        # [CẢI TIẾN]: Lưu mô hình dựa trên QWK cao nhất thay vì MSE thấp nhất
        best_val_qwk = -float('inf') 
        best_state = None
        no_improve = 0
        criterion = nn.SmoothL1Loss(beta=1.0, reduction='none')
        
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels         = batch["labels"].to(DEVICE)
                
                optimizer.zero_grad()
                
                # [CẢI TIẾN]: Chạy Forward pass với Autocast (tiết kiệm bộ nhớ, tăng tốc độ)
                with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                    preds = model(input_ids, attention_mask)
                    raw_loss = criterion(preds, labels)
                    
                    label_indices = (labels.long() - 1).clamp(0, 4)
                    sample_weights = self.class_weights[label_indices]
                    loss = (raw_loss * sample_weights).mean()

                # Backward pass dùng Scaler
                scaler.scale(loss).backward()
                
                # Unscale trước khi clip norm
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            val_preds, val_labels = self._predict_loader(model, val_loader)
            val_loss = np.mean((val_preds - val_labels)**2)
            
            epoch_rounder = OptimizedRounder()
            epoch_rounder.fit(val_preds, val_labels, verbose=False)
            epoch_preds = epoch_rounder.predict(val_preds, epoch_rounder.coef_)
            val_qwk_opt = cohen_kappa_score(val_labels, epoch_preds, weights="quadratic")

            # [CẢI TIẾN]: Quyết định lưu theo val_qwk_opt
            flag = ("✔ best" if val_qwk_opt > best_val_qwk 
                    else f"(no improve {no_improve+1}/{self.early_stopping})")
            
            print(f"  Epoch {epoch+1:>2}/{self.epochs} "
                  f"train_loss={avg_loss:.4f} "
                  f"val_mse={val_loss:.4f} "
                  f"val_qwk_opt={val_qwk_opt:.4f} " 
                  f"{flag}")

            if val_qwk_opt > best_val_qwk:
                best_val_qwk, no_improve = val_qwk_opt, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= self.early_stopping:
                    print(f"  ⏹ Early stopping tại epoch {epoch+1}")
                    break

        model.load_state_dict(best_state)
        return model, best_val_qwk

    @staticmethod
    def _predict_loader(model, loader) -> tuple:
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                
                # Inference với AMP
                with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                    preds = model(input_ids, attention_mask)
                    
                all_preds.extend(preds.cpu().float().numpy())
                if "labels" in batch:
                    all_labels.extend(batch["labels"].numpy())
        return np.array(all_preds), np.array(all_labels)

    def train(self, df: pd.DataFrame, label_col: str = "label"):
        texts  = self.build_texts(df)
        labels = df[label_col].values

        self.class_weights = compute_class_weights(labels, 5)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        self.oof_raw_preds = np.zeros(len(df), dtype=float)

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

            model, _ = self._train_fold(train_loader, val_loader, fold)
            self.models.append(model)

            val_preds, _ = self._predict_loader(model, val_loader)
            self.oof_raw_preds[val_idx] = val_preds

        print(f"\n{'='*50}")
        print("Đang tối ưu hóa ngưỡng cắt toàn cục (Global Threshold Optimization)...")
        self.rounder.fit(self.oof_raw_preds, labels)
        
        final_oof_preds = self.rounder.predict(self.oof_raw_preds, self.rounder.coef_)
        oof_qwk = cohen_kappa_score(labels, final_oof_preds, weights="quadratic")
        oof_f1  = f1_score(labels, final_oof_preds, average="macro")
        
        print(f"Overall OOF QWK     : {oof_qwk:.4f}")
        print(f"Overall OOF Macro F1: {oof_f1:.4f}")
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
        
        all_preds = np.zeros(len(texts))
        for model in self.models:
            model.eval()
            model.to(DEVICE)
            fold_preds = []
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch["input_ids"].to(DEVICE)
                    attention_mask = batch["attention_mask"].to(DEVICE)
                    
                    with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                        preds = model(input_ids, attention_mask)
                        
                    fold_preds.extend(preds.cpu().float().numpy())
            all_preds += np.array(fold_preds)
            
        all_preds /= len(self.models)
        
        return self.rounder.predict(all_preds, self.rounder.coef_)

    def save_models(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        for i, model in enumerate(self.models):
            path = os.path.join(save_dir, f"bert_fold_{i+1}.pt")
            torch.save(model.state_dict(), path)
            print(f"Saved: {path}")
        np.save(os.path.join(save_dir, "rounder_coefs.npy"), self.rounder.coef_)
            
    def load_models(self, save_dir: str):
        self.models = []
        for i in range(self.n_splits):
            path = os.path.join(save_dir, f"bert_fold_{i+1}.pt")
            model = BERTOrdinalRegressor(model_name=self.model_name)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.to(DEVICE)
            self.models.append(model)
            print(f"Loaded: {path}")
        coef_path = os.path.join(save_dir, "rounder_coefs.npy")
        if os.path.exists(coef_path):
            self.rounder.coef_ = np.load(coef_path)
            print(f"Loaded Thresholds: {self.rounder.coef_}")