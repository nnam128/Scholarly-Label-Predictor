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

#  [0] Device & Settings 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
NUM_WORKERS = 0 if DEVICE.type == "cpu" else 2


#  [1] Tiện ích: Class Weights & FGM 
def compute_class_weights(labels: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    counts  = np.bincount(labels - 1, minlength=num_classes).astype(float)
    weights = len(labels) / (num_classes * counts)
    weights = weights / weights.mean()
    print(f"Class weights: { {i+1: round(w,3) for i,w in enumerate(weights)} }")
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


class FGM:
    """Fast Gradient Method - Bơm nhiễu vào lớp Embedding để chống Overfitting"""
    def __init__(self, model):
        self.model  = model
        self.backup = {}

    def attack(self, epsilon=0.2, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


#  [2] Optimized Rounder (tối ưu đồng thời QWK + Macro F1) 
class OptimizedRounder:
    def __init__(self):
        self.coef_ = np.array([1.5, 2.5, 3.5, 4.5])

    def _predict_from_coef(self, coef, X):
        coef = np.sort(coef)
        X_p  = np.copy(X)
        for i, pred in enumerate(X_p):
            if   pred < coef[0]: X_p[i] = 1
            elif pred < coef[1]: X_p[i] = 2
            elif pred < coef[2]: X_p[i] = 3
            elif pred < coef[3]: X_p[i] = 4
            else:                X_p[i] = 5
        return X_p.astype(int)

    def _kappa_loss(self, coef, X, y):
        X_p = self._predict_from_coef(coef, X)
        return -cohen_kappa_score(y, X_p, weights='quadratic')

    def _combined_loss(self, coef, X, y, f1_weight=0.4):
        """Tối ưu tổ hợp QWK + Macro F1 — tune f1_weight nếu cần."""
        X_p = self._predict_from_coef(coef, X)
        qwk = cohen_kappa_score(y, X_p, weights='quadratic')
        f1  = f1_score(y, X_p, average='macro')
        return -((1 - f1_weight) * qwk + f1_weight * f1)

    def fit(self, X, y, f1_weight=0.4, verbose=True):
        loss_fn = partial(self._combined_loss, X=X, y=y, f1_weight=f1_weight)
        initial_coef = [1.5, 2.5, 3.5, 4.5]
        result = sp_opt.minimize(loss_fn, initial_coef, method='nelder-mead')
        self.coef_ = np.sort(result.x)
        if verbose:
            preds = self._predict_from_coef(self.coef_, X)
            qwk = cohen_kappa_score(y, preds, weights='quadratic')
            f1  = f1_score(y, preds, average='macro')
            print(f"Optimized Thresholds : {np.round(self.coef_, 4)}")
            print(f"  → QWK={qwk:.4f}  Macro-F1={f1:.4f}  (f1_weight={f1_weight})")

    def predict(self, X, coef=None):
        if coef is None:
            coef = self.coef_
        return self._predict_from_coef(coef, X)


#  [3] Dataset (Ordinal Frank-Hall targets) 
class PaperDataset(Dataset):
    def __init__(self, texts: list, labels: Optional[list], tokenizer, max_length: int = 512):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
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
            raw_label = self.labels[idx]
            item["raw_label"] = torch.tensor(raw_label, dtype=torch.long)
            ordinal_target    = [1.0 if raw_label > i else 0.0 for i in range(1, 5)]
            item["labels"]    = torch.tensor(ordinal_target, dtype=torch.float32)
        return item


def dynamic_collate_fn(batch):
    input_ids      = nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=0)
    attention_mask = nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    result = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "labels" in batch[0]:
        result["labels"]    = torch.stack([b["labels"]    for b in batch])
        result["raw_label"] = torch.stack([b["raw_label"] for b in batch])
    return result


#  [4] Model: SciBERT Dual-Head (Ordinal + CE) 
class BERTOrdinalRegressor(nn.Module):
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased"):
        super().__init__()
        self.bert     = AutoModel.from_pretrained(model_name)
        hidden        = self.bert.config.hidden_size
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)])

        # Ordinal head: 4 node (Frank-Hall encoding)
        self.fc_ordinal = nn.Linear(hidden * 2, 4)
        # CE head: 5 node (cross-entropy classification)
        self.fc_ce      = nn.Linear(hidden * 2, 5)

    def forward(self, input_ids, attention_mask):
        out       = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = out.last_hidden_state

        cls_token   = token_emb[:, 0, :]
        mask        = attention_mask.unsqueeze(-1).float()
        mean_pooled = (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        x           = torch.cat((cls_token, mean_pooled), dim=1)

        # Multi-dropout ensemble cho cả hai head
        ord_out = torch.mean(
            torch.stack([self.fc_ordinal(d(x)) for d in self.dropouts]), dim=0)
        ce_out  = torch.mean(
            torch.stack([self.fc_ce(d(x))      for d in self.dropouts]), dim=0)

        return ord_out, ce_out


#  [5] LLRD Optimizer 
def make_llrd_optimizer(model: BERTOrdinalRegressor, base_lr: float, decay: float = 0.95):
    no_decay   = ["bias", "LayerNorm.weight"]
    num_layers = model.bert.config.num_hidden_layers
    param_groups = []

    # Classifier heads — full lr
    for head_name in ["fc_ordinal", "fc_ce"]:
        head = getattr(model, head_name)
        param_groups += [
            {"params": [p for n, p in head.named_parameters() if not any(nd in n for nd in no_decay)],
             "lr": base_lr, "weight_decay": 0.01},
            {"params": [p for n, p in head.named_parameters() if     any(nd in n for nd in no_decay)],
             "lr": base_lr, "weight_decay": 0.0},
        ]

    # Transformer layers — LLRD
    for layer_idx in range(num_layers - 1, -1, -1):
        layer_lr = base_lr * (decay ** (num_layers - layer_idx))
        layer    = model.bert.encoder.layer[layer_idx]
        param_groups += [
            {"params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
             "lr": layer_lr, "weight_decay": 0.01},
            {"params": [p for n, p in layer.named_parameters() if     any(nd in n for nd in no_decay)],
             "lr": layer_lr, "weight_decay": 0.0},
        ]

    # Embeddings — lowest lr
    embed_lr = base_lr * (decay ** (num_layers + 1))
    param_groups += [
        {"params": [p for n, p in model.bert.embeddings.named_parameters() if not any(nd in n for nd in no_decay)],
         "lr": embed_lr, "weight_decay": 0.01},
        {"params": [p for n, p in model.bert.embeddings.named_parameters() if     any(nd in n for nd in no_decay)],
         "lr": embed_lr, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(param_groups)


#  [6] BERTTrainer 
class BERTTrainer:
    def __init__(
        self,
        model_name:     str   = "allenai/scibert_scivocab_uncased",
        max_length:     int   = 256,
        epochs:         int   = 10,
        early_stopping: int   = 3,
        batch_size:     int   = 8,
        lr:             float = 2e-5,
        n_splits:       int   = 5,
        llrd_decay:     float = 0.95,
        alpha:          float = 0.7,
        beta:           float = 0.3,
        gamma:          float = 0.6,
        f1_weight:      float = 0.4,
    ):
        self.model_name     = model_name
        self.max_length     = max_length
        self.epochs         = epochs
        self.early_stopping = early_stopping
        self.batch_size     = batch_size
        self.lr             = lr
        self.n_splits       = n_splits
        self.llrd_decay     = llrd_decay
        self.alpha          = alpha
        self.beta           = beta
        self.gamma          = gamma
        self.f1_weight      = f1_weight

        self.tokenizer     = AutoTokenizer.from_pretrained(model_name)
        self.models        = []
        self.oof_raw_preds = None
        self.class_weights = None
        self.rounder       = OptimizedRounder()

    def build_texts(self, df: pd.DataFrame) -> list:
        texts = []
        sep   = self.tokenizer.sep_token if self.tokenizer.sep_token else "[SEP]"
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
                f"{title} {sep} {abstract} {sep} year: {year}, authors: {authors_short}"
            )
        return texts

    def _train_fold(self, train_loader, val_loader, fold: int):
        model     = BERTOrdinalRegressor(self.model_name).to(DEVICE)
        optimizer = make_llrd_optimizer(model, self.lr, self.llrd_decay)
        fgm       = FGM(model)

        total_steps = len(train_loader) * self.epochs
        scheduler   = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, total_steps=total_steps, pct_start=0.1,
        )

        criterion_bce = nn.BCEWithLogitsLoss(reduction='none')
        criterion_ce  = nn.CrossEntropyLoss(reduction='none')

        best_val_qwk = -float('inf')
        best_state   = None
        no_improve   = 0

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels         = batch["labels"].to(DEVICE)       # ordinal targets
                raw_labels     = batch["raw_label"].to(DEVICE)
                ce_targets     = (raw_labels.long() - 1).clamp(0, 4)  # 0-indexed
                sample_weights = self.class_weights[ce_targets]

                optimizer.zero_grad()

                # --- Forward chuẩn ---
                logits_ord, logits_ce = model(input_ids, attention_mask)

                loss_ord = (criterion_bce(logits_ord, labels).mean(dim=1) * sample_weights).mean()
                loss_ce  = (criterion_ce(logits_ce, ce_targets)           * sample_weights).mean()
                loss     = self.alpha * loss_ord + self.beta * loss_ce

                loss.backward()

                # --- Adversarial Attack ---
                fgm.attack(epsilon=0.2, emb_name='word_embeddings')
                logits_ord_adv, logits_ce_adv = model(input_ids, attention_mask)
                loss_ord_adv = (criterion_bce(logits_ord_adv, labels).mean(dim=1) * sample_weights).mean()
                loss_ce_adv  = (criterion_ce(logits_ce_adv, ce_targets)           * sample_weights).mean()
                loss_adv     = self.alpha * loss_ord_adv + self.beta * loss_ce_adv
                loss_adv.backward()
                fgm.restore()

                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            #  Validation 
            val_preds, val_labels = self._predict_loader(model, val_loader, self.gamma)
            epoch_rounder = OptimizedRounder()
            epoch_rounder.fit(val_preds, val_labels, f1_weight=self.f1_weight, verbose=False)
            epoch_preds  = epoch_rounder.predict(val_preds)
            val_qwk_opt  = cohen_kappa_score(val_labels, epoch_preds, weights="quadratic")
            val_f1_opt   = f1_score(val_labels, epoch_preds, average="macro")

            flag = ("✔ best" if val_qwk_opt > best_val_qwk
                    else f"(no improve {no_improve+1}/{self.early_stopping})")
            print(f"  Epoch {epoch+1:>2}/{self.epochs} "
                  f"loss={avg_loss:.4f} "
                  f"qwk={val_qwk_opt:.4f} f1={val_f1_opt:.4f} {flag}")

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
    def _predict_loader(model, loader, gamma: float = 0.6) -> tuple:
        """
        gamma: trọng số cho ordinal head (0.6 mặc định)
               Giảm gamma → nghiêng về CE head → F1 tăng thêm
        """
        model.eval()
        all_preds, all_labels = [], []
        classes = torch.arange(1, 6, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)

                logits_ord, logits_ce = model(input_ids, attention_mask)

                # Ordinal expected value ∈ [1, 5]
                ev_ord = 1.0 + torch.sigmoid(logits_ord).sum(dim=1)

                # CE expected value ∈ [1, 5]
                ev_ce  = (torch.softmax(logits_ce, dim=1) * classes).sum(dim=1)

                # Blend
                ev_blend = gamma * ev_ord + (1 - gamma) * ev_ce

                all_preds.extend(ev_blend.cpu().float().numpy())
                if "raw_label" in batch:
                    all_labels.extend(batch["raw_label"].numpy())

        return np.array(all_preds), np.array(all_labels)

    def train(self, df: pd.DataFrame, label_col: str = "label"):
        texts  = self.build_texts(df)
        labels = df[label_col].values

        self.class_weights = compute_class_weights(labels, 5)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        self.oof_raw_preds = np.zeros(len(df), dtype=float)

        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
            print(f"\n{'='*55}")
            print(f"Fold {fold+1}/{self.n_splits}")
            print(f"{'='*55}")

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

            val_preds, _ = self._predict_loader(model, val_loader, self.gamma)
            self.oof_raw_preds[val_idx] = val_preds

        print(f"\n{'='*55}")
        print("Tối ưu ngưỡng cắt toàn cục (QWK + Macro F1)...")
        self.rounder.fit(self.oof_raw_preds, labels, f1_weight=self.f1_weight)

        final_preds = self.rounder.predict(self.oof_raw_preds)
        oof_qwk = cohen_kappa_score(labels, final_preds, weights="quadratic")
        oof_f1  = f1_score(labels, final_preds, average="macro")
        print(f"Overall OOF QWK     : {oof_qwk:.4f}")
        print(f"Overall OOF Macro F1: {oof_f1:.4f}")
        print(f"{'='*55}")
        return oof_qwk

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        texts   = self.build_texts(df)
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
                    input_ids      = batch["input_ids"].to(DEVICE)
                    attention_mask = batch["attention_mask"].to(DEVICE)
                    logits_ord, logits_ce = model(input_ids, attention_mask)
                    ev_ord   = 1.0 + torch.sigmoid(logits_ord).sum(dim=1)
                    classes  = torch.arange(1, 6, dtype=torch.float32, device=DEVICE)
                    ev_ce    = (torch.softmax(logits_ce, dim=1) * classes).sum(dim=1)
                    ev_blend = self.gamma * ev_ord + (1 - self.gamma) * ev_ce
                    fold_preds.extend(ev_blend.cpu().float().numpy())
            all_preds += np.array(fold_preds)

        all_preds /= len(self.models)
        return self.rounder.predict(all_preds)

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
            path  = os.path.join(save_dir, f"bert_fold_{i+1}.pt")
            model = BERTOrdinalRegressor(model_name=self.model_name)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.to(DEVICE)
            self.models.append(model)
            print(f"Loaded: {path}")
        coef_path = os.path.join(save_dir, "rounder_coefs.npy")
        if os.path.exists(coef_path):
            self.rounder.coef_ = np.load(coef_path)
            print(f"Loaded Thresholds: {self.rounder.coef_}")