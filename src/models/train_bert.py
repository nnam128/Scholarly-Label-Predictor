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
warnings.filterwarnings("ignore")


#  Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
NUM_WORKERS = 0 if DEVICE.type == "cpu" else 2


# ── [4] Class weights helper ───────────────────────────────────────────────────
def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency weights, normalize về mean=1.
    Label 1 (nhiều) → weight thấp, Label 5 (ít) → weight cao.
    Truyền vào CrossEntropy để bù imbalance 3.3x.
    """
    counts  = np.bincount(labels - 1, minlength=num_classes).astype(float)
    weights = len(labels) / (num_classes * counts)
    weights = weights / weights.mean()
    print(f"Class weights: { {i+1: round(w,3) for i,w in enumerate(weights)} }")
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


# ── [1] QWK Surrogate Loss ─────────────────────────────────────────────────────
def qwk_surrogate_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 5,
    alpha: float = 0.5,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    QWK Surrogate = CrossEntropy + Expected-rank penalty có trọng số bậc 2.

    QWK phạt sai lệch theo bậc 2: sai 2 bậc nặng gấp 4 lần sai 1 bậc.
    Penalty dưới đây mô phỏng đúng tính chất này:
        penalty = E[((pred_rank - true_rank) / (num_classes-1))^2]
               = sum_k p_k * ((k - y) / (C-1))^2

    So với ordinal_distance_loss dùng |k-y| tuyến tính,
    QWK surrogate dùng (k-y)^2 bậc 2 — đúng với công thức QWK gốc.

    class_weights: truyền vào CE để xử lý imbalance.
    alpha=0.5: cân bằng CE và QWK penalty.
    """
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)(logits, labels)

    probs  = torch.softmax(logits, dim=1)                          # (B, C)
    ranks  = torch.arange(num_classes, device=logits.device,
                          dtype=torch.float32)                     # (C,)
    y      = labels.float()                                        # (B,)
    scale  = (num_classes - 1) ** 2                               # normalize như QWK

    # (B, C): bình phương khoảng cách từ mỗi rank đến true label
    sq_dist = ((ranks.unsqueeze(0) - y.unsqueeze(1)) ** 2) / scale
    penalty = (probs * sq_dist).sum(dim=1).mean()

    return ce_loss + alpha * penalty


#  1. Loss: R-Drop (giữ nguyên, chỉ đổi inner loss sang QWK surrogate)
def rdrop_loss(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 5,
    alpha: float = 0.5,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    R-Drop với QWK surrogate loss bên trong.
    loss = 0.5*(QWK1 + QWK2) + alpha * KL(p1 || p2)
    """
    ce1 = qwk_surrogate_loss(logits1, labels, num_classes, alpha=0.5,
                             class_weights=class_weights)
    ce2 = qwk_surrogate_loss(logits2, labels, num_classes, alpha=0.5,
                             class_weights=class_weights)
    ce  = 0.5 * (ce1 + ce2)

    p1  = torch.softmax(logits1, dim=1)
    p2  = torch.softmax(logits2, dim=1)
    kl1 = nn.functional.kl_div(torch.log(p1 + 1e-8), p2, reduction="batchmean")
    kl2 = nn.functional.kl_div(torch.log(p2 + 1e-8), p1, reduction="batchmean")
    kl  = 0.5 * (kl1 + kl2)

    return ce + alpha * kl


#  3. Dataset với Dynamic Padding
class PaperDataset(Dataset):
    """
    Dynamic padding: không pad về max_length cố định mà pad về
    max length trong batch → giảm ~30% thời gian train vì ít token padding hơn.
    """
    def __init__(
        self,
        texts: list,
        labels: Optional[list],
        tokenizer,
        max_length: int = 256,
    ):
        self.texts = texts
        self.labels = labels  # None khi predict
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
            item["labels"] = torch.tensor(self.labels[idx] - 1, dtype=torch.long)  # 1-5 → 0-4
        return item


def dynamic_collate_fn(batch):
    """
    Pad về max length của batch hiện tại thay vì max_length cố định.
    Batch có câu ngắn → ít token padding → forward/backward nhanh hơn.
    """
    input_ids      = [item["input_ids"]      for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]

    # Pad về max len trong batch này
    input_ids      = nn.utils.rnn.pad_sequence(input_ids,      batch_first=True, padding_value=0)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    result = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "labels" in batch[0]:
        result["labels"] = torch.stack([item["labels"] for item in batch])
    return result


# ── [2] Model với Mean Pooling ─────────────────────────────────────────────────
class BERTOrdinalClassifier(nn.Module):
    """
    Mean pooling thay CLS pooling.

    Tại sao mean pooling tốt hơn cho scientific text:
    - CLS token được train cho NSP task (không có trong SciBERT) → representation kém
    - Mean pooling lấy trung bình tất cả token (trừ padding) → capture toàn bộ context
    - Với title-only text (~20-40 token), mean pooling ổn định hơn CLS

    attention_mask dùng để mask padding token trước khi mean.
    """
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased",
                 num_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size  # 768
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # [2] Mean pooling — trung bình các token thực (bỏ padding)
        token_emb = out.last_hidden_state                          # (B, L, H)
        mask      = attention_mask.unsqueeze(-1).float()           # (B, L, 1)
        sum_emb   = (token_emb * mask).sum(dim=1)                  # (B, H)
        count     = mask.sum(dim=1).clamp(min=1e-9)               # (B, 1)
        pooled    = sum_emb / count                                # (B, H)

        return self.classifier(pooled)


#  5. LLRD Optimizer
def make_llrd_optimizer(model: BERTOrdinalClassifier, base_lr: float,
                        decay: float = 0.9) -> torch.optim.AdamW:
    """
    Layer-wise Learning Rate Decay.

    Tại sao hiệu quả:
    - Layer đầu BERT học low-level features (syntax) → đã pretrain tốt → lr nhỏ
    - Layer cuối học high-level features (semantics) → cần fine-tune nhiều hơn → lr lớn
    - Classifier head hoàn toàn mới → lr lớn nhất

    decay=0.9: mỗi layer giảm 10% lr so với layer trên.
    Layer 0 (embedding): base_lr * 0.9^12 ≈ base_lr * 0.28
    Layer 11 (top BERT): base_lr * 0.9^1  ≈ base_lr * 0.90
    Classifier head   : base_lr * 1.0
    """
    num_layers = model.bert.config.num_hidden_layers  # 12 với BERT-base

    # Classifier head — lr cao nhất
    param_groups = [
        {"params": model.classifier.parameters(), "lr": base_lr}
    ]

    # BERT layers — lr giảm dần từ trên xuống
    for layer_idx in range(num_layers - 1, -1, -1):
        layer_lr = base_lr * (decay ** (num_layers - layer_idx))
        param_groups.append({
            "params": model.bert.encoder.layer[layer_idx].parameters(),
            "lr": layer_lr,
        })

    # Embeddings — lr thấp nhất
    embed_lr = base_lr * (decay ** (num_layers + 1))
    param_groups.append({
        "params": model.bert.embeddings.parameters(),
        "lr": embed_lr,
    })

    return torch.optim.AdamW(param_groups, weight_decay=0.01)


# ── [3] Expected-rank decoding ─────────────────────────────────────────────────
def expected_rank_decode(probs: np.ndarray, num_classes: int = 5) -> np.ndarray:
    """
    Thay argmax bằng kỳ vọng của phân phối rồi làm tròn.

    argmax chọn class có prob cao nhất → bỏ qua thông tin ordinal.
    Expected rank = sum_k (k * p_k) → dùng toàn bộ distribution.

    Ví dụ: probs = [0.1, 0.4, 0.4, 0.1, 0.0]
        argmax       → class 1 hoặc 2 (tie, lấy đầu tiên)
        expected rank → 0.1*0 + 0.4*1 + 0.4*2 + 0.1*3 = 1.5 → round → 2

    Đặc biệt hữu ích khi probability mass trải đều giữa 2 class liền kề.
    Trả về label 1-5 (0-based rank + 1).
    """
    ranks = np.arange(num_classes)                    # [0,1,2,3,4]
    exp   = (probs * ranks).sum(axis=1)               # (N,) float
    return np.clip(np.round(exp).astype(int), 0, num_classes - 1) + 1  # 1-5


#  6. Trainer
class BERTTrainer:
    """
    Fine-tune SciBERT với các kỹ thuật:
    [A] R-Drop Regularization
    [B] LLRD Optimizer
    [C] QWK Surrogate Loss  ← đổi từ ordinal_distance_loss
    [D] Dynamic Padding
    [E] Mean Pooling         ← đổi từ CLS pooling
    [F] Class Weights        ← thêm mới
    [G] Expected-rank decode ← đổi từ argmax

    Val metric: QWK (cohen_kappa_score quadratic) thay vì Macro F1
    vì leaderboard dùng QWK.
    """

    def __init__(
        self,
        model_name:     str   = "allenai/scibert_scivocab_uncased",
        num_classes:    int   = 5,
        max_length:     int   = 256,
        epochs:         int   = 17,
        early_stopping: int   = 3,
        batch_size:     int   = 8,
        lr:             float = 2e-5,
        n_splits:       int   = 5,
        alpha:          float = 0.5,
        llrd_decay:     float = 0.9,
        use_rdrop:      bool  = True,
    ):
        self.model_name     = model_name
        self.num_classes    = num_classes
        self.max_length     = max_length
        self.epochs         = epochs
        self.early_stopping = early_stopping
        self.batch_size     = batch_size
        self.lr             = lr
        self.n_splits       = n_splits
        self.alpha          = alpha
        self.llrd_decay     = llrd_decay
        self.use_rdrop      = use_rdrop

        self.tokenizer     = AutoTokenizer.from_pretrained(model_name)
        self.models        = []
        self.oof_preds     = None
        self.class_weights = None   # set trong train()

    @staticmethod
    def build_texts(df: pd.DataFrame) -> list:
        texts = []
        for _, row in df.iterrows():
            title    = str(row.get("title",    "")).strip()
            abstract = str(row.get("abstract", "")).strip()
            venue    = str(row.get("venue",    "")).strip()
            year     = str(row.get("year",     "")).strip()
            authors  = str(row.get("authors",  "")).strip()
            
            # Chỉ lấy 3 tác giả đầu để tiết kiệm token
            author_list   = [a.strip() for a in authors.split(",") if a.strip()]
            authors_short = ", ".join(author_list[:3])
            if len(author_list) > 3:
                authors_short += " et al."
                
            texts.append(
                #f"venue: {venue} "
                f"year: {year} "
                f"authors: {authors_short} "
                f"title: {title} "
                #f"[SEP] abstract: {abstract}"
            )
        return texts
    
    #  train 1 fold 
    def _train_fold(self, train_loader, val_loader, fold: int):
        model     = BERTOrdinalClassifier(self.model_name, self.num_classes).to(DEVICE)
        optimizer = make_llrd_optimizer(model, self.lr, self.llrd_decay)  # [B] LLRD

        total_steps  = len(train_loader) * self.epochs
        scheduler    = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, total_steps=total_steps, pct_start=0.1,
        )

        best_qwk   = -1
        best_state = None
        no_improve = 0
        
        for epoch in range(self.epochs):
            #  train 
            model.train()
            total_loss = 0
            for batch in train_loader:
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels         = batch["labels"].to(DEVICE)
                
                optimizer.zero_grad()

                if self.use_rdrop:
                    logits1 = model(input_ids, attention_mask)
                    logits2 = model(input_ids, attention_mask)
                    loss    = rdrop_loss(logits1, logits2, labels,
                                        self.num_classes, self.alpha,
                                        self.class_weights)
                else:
                    logits = model(input_ids, attention_mask)
                    loss   = qwk_surrogate_loss(logits, labels,
                                                self.num_classes, self.alpha,
                                                self.class_weights)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # [G] Validate bằng QWK + expected-rank decode
            val_preds, val_labels = self._predict_loader(model, val_loader)
            val_qwk = cohen_kappa_score(val_labels + 1, val_preds,
                                        weights="quadratic")
            val_f1  = f1_score(val_labels + 1, val_preds, average="macro")

            flag = ("✔ best" if val_qwk > best_qwk
                    else f"(no improve {no_improve+1}/{self.early_stopping})")
            print(f"  Epoch {epoch+1:>2}/{self.epochs} "
                  f"loss={avg_loss:.4f} "
                  f"val_qwk={val_qwk:.4f} "
                  f"val_f1={val_f1:.4f} "
                  f"{flag}")

            if val_qwk > best_qwk:
                best_qwk, no_improve = val_qwk, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= self.early_stopping:
                    print(f"  ⏹ Early stopping tại epoch {epoch+1}")
                    break

        model.load_state_dict(best_state)
        return model, best_qwk

    @staticmethod
    def _predict_loader(model, loader) -> tuple:
        """
        Trả về (preds_1to5, labels_0based).
        preds dùng expected-rank decode thay argmax.
        """
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                logits = model(input_ids, attention_mask)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                if "labels" in batch:
                    all_labels.extend(batch["labels"].numpy())

        all_probs = np.vstack(all_probs)                           # (N, C)
        preds     = expected_rank_decode(all_probs, model.classifier[-1].out_features)
        return preds, np.array(all_labels)

    def train(self, df: pd.DataFrame, label_col: str = "label"):
        """
        Parameters
        ----------
        df        : DataFrame có cột title, abstract, venue, year, authors, label (1-5).
        label_col : Tên cột nhãn.
        """
        texts  = self.build_texts(df)
        labels = df[label_col].values

        # [4] Tính class weights từ toàn bộ train set
        self.class_weights = compute_class_weights(labels, self.num_classes)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        self.oof_preds = np.zeros(len(df), dtype=int)

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

            model, best_qwk = self._train_fold(train_loader, val_loader, fold)
            print(f"  → Best val QWK: {best_qwk:.4f}")
            self.models.append(model)

            val_preds, _ = self._predict_loader(model, val_loader)
            self.oof_preds[val_idx] = val_preds

        oof_qwk = cohen_kappa_score(labels, self.oof_preds, weights="quadratic")
        oof_f1  = f1_score(labels, self.oof_preds, average="macro")
        print(f"\n{'='*50}")
        print(f"Overall OOF QWK    : {oof_qwk:.4f}")
        print(f"Overall OOF Macro F1: {oof_f1:.4f}")
        print(f"{'='*50}")
        return oof_qwk

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Ensemble predict: average softmax prob từ tất cả fold models.
        Trả về nhãn 1-5.
        """
        texts = self.build_texts(df)
        test_ds = PaperDataset(texts, None, self.tokenizer, self.max_length)
        loader  = DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
            collate_fn=dynamic_collate_fn,
        )
        all_probs = np.zeros((len(texts), self.num_classes))
        for model in self.models:
            model.eval()
            model.to(DEVICE)
            fold_probs = []
            with torch.no_grad():
                for batch in loader:
                    logits = model(batch["input_ids"].to(DEVICE),
                                   batch["attention_mask"].to(DEVICE))
                    fold_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_probs += np.vstack(fold_probs)
        all_probs /= len(self.models)
        # [G] Expected-rank decode trên ensemble probs
        return expected_rank_decode(all_probs, self.num_classes)

    def save_models(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        for i, model in enumerate(self.models):
            path = os.path.join(save_dir, f"bert_fold_{i+1}.pt")
            torch.save(model.state_dict(), path)
            print(f"Saved: {path}")
            
    def load_models(self, save_dir: str):
        self.models = []
        for i in range(self.n_splits):
            path = os.path.join(save_dir, f"bert_fold_{i+1}.pt")
            model = BERTOrdinalClassifier(
                model_name=self.model_name,
                num_classes=self.num_classes,
            )
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.to(DEVICE)
            self.models.append(model)
            print(f"Loaded: {path}")