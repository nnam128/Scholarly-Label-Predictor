import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
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

# num_workers=0 trên CPU để tránh fork/deadlock, 2 khi có GPU
NUM_WORKERS = 0 if DEVICE.type == "cpu" else 2


# 1. Ordinal Distance Loss
def ordinal_distance_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 5,
    alpha: float = 0.75,
    class_weights: torch.Tensor = None,
) -> torch.Tensor:
    """
    CrossEntropy + Ordinal distance penalty.
    Predict gần label thật bị phạt nhẹ hơn predict xa.
    """
    #  CrossEntropy chuẩn 
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    ce_loss = ce_loss_fn(logits, labels)
    
    #  Ordinal penalty
    probs = torch.softmax(logits, dim=1)
    
    ranks = torch.arange(
        num_classes,
        device=logits.device,
        dtype=torch.float32
    )
    
    labels_expanded = labels.unsqueeze(1).float()
    
    # |pred_class - true_class|
    distances = torch.abs(ranks.unsqueeze(0) - labels_expanded)
    
    # expected ordinal distance
    expected_distance = (probs * distances).sum(dim=1).mean()
    
    loss = ce_loss + alpha * expected_distance
    
    return loss


#  2. Dataset
class PaperDataset(Dataset):
    """
    Nhận list texts và labels (tuỳ chọn).
    Label ordinal: 1-5 → shift về 0-4.
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
            padding="max_length",
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


#  3. Model
class BERTOrdinalClassifier(nn.Module):
    """
    SciBERT + dropout + linear head.
    Loss = CrossEntropy, predict bằng argmax → 0-4 → +1 → 1-5.
    """
    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        num_classes: int = 5,
        dropout: float = 0.2,
    ):
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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.classifier(cls)


#  4. Trainer
class BERTTrainer:
    """
    Fine-tune SciBERT cho bài ordinal classification (label 1-5).
    Parameters
    ----------
    model_name     : HuggingFace model id. Mặc định SciBERT.
    num_classes    : Số class (5).
    max_length     : Max token length. 256
    epochs         : Số epoch tối đa mỗi fold (default 20).
    early_stopping : Dừng nếu val F1 không cải thiện sau N epoch liên tiếp (default 4).
    batch_size     : 16 an toàn trên T4; giảm xuống 8 nếu OOM.
    lr             : Learning rate. 2e-5 là sweet spot cho fine-tune BERT.
    n_splits       : Số fold CV.
    """
    
    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        num_classes: int = 5,
        max_length: int = 256,
        epochs: int = 17,
        early_stopping: int = 3,
        batch_size: int = 8,
        lr: float = 2e-5,
        n_splits: int = 5,
        alpha: float = 0.5,
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.batch_size = batch_size
        self.lr = lr
        self.n_splits = n_splits
        self.alpha = alpha
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.models = []
        self.oof_preds = None
        
    #  Helper: build text 
    @staticmethod
    def build_texts(df: pd.DataFrame) -> list:
        """
        Ghép tất cả cột có ý nghĩa thành 1 chuỗi đầu vào cho BERT.
        Format:
            venue: <v> year: <y> authors: <a> <title> ( <v> ) [SEP] <abstract>
        Abstract dài đặt cuối — truncation cắt ở đây nếu quá max_length.
        """
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
        model = BERTOrdinalClassifier(
            model_name=self.model_name,
            num_classes=self.num_classes,
        ).to(DEVICE)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.01) 
        
        # OneCycleLR: warmup 10% → cosine decay
        total_steps  = len(train_loader) * self.epochs
        warmup_steps = int(0.1 * total_steps)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
        )

        best_f1 = -1
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
                logits = model(input_ids, attention_mask)

                #  NEW LOSS 
                loss = ordinal_distance_loss(
                    logits,
                    labels,
                    num_classes=self.num_classes,
                    alpha=self.alpha,
                )

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            
            #  validate 
            val_preds, val_labels = self._predict_loader(
                model,
                val_loader
            )

            val_f1 = f1_score(
                val_labels + 1,
                val_preds + 1,
                average="macro"
            )

            flag = (
                "✔ best"
                if val_f1 > best_f1
                else f"(no improve {no_improve+1}/{self.early_stopping})"
            )

            print(
                f"  Epoch {epoch+1:>2}/{self.epochs} "
                f"loss={avg_loss:.4f} "
                f"val_macro_f1={val_f1:.4f} "
                f"{flag}"
            )

            if val_f1 > best_f1:
                best_f1    = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.early_stopping:
                    print(f"  ⏹ Early stopping tại epoch {epoch+1}")
                    break
                
        model.load_state_dict(best_state)
        return model, best_f1
    
    #  Predict từ DataLoader 
    @staticmethod
    def _predict_loader(model, loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                logits = model(input_ids, attention_mask)
                preds  = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                if "labels" in batch:
                    all_labels.extend(batch["labels"].numpy())
        return np.array(all_preds), np.array(all_labels)
    
    #  Public: train với Stratified K-Fold CV 
    def train(self, df: pd.DataFrame, label_col: str = "label"):
        """
        Parameters
        ----------
        df        : DataFrame có cột title, abstract, venue, year, authors, label (1-5).
        label_col : Tên cột nhãn.
        """
        texts  = self.build_texts(df)
        labels = df[label_col].values
        
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
            )
            val_loader = DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
            )
            
            model, best_f1 = self._train_fold(train_loader, val_loader, fold)
            print(f"  → Best val Macro F1: {best_f1:.4f}")
            self.models.append(model)
            
            val_preds, _ = self._predict_loader(model, val_loader)
            self.oof_preds[val_idx] = val_preds + 1  # 0-4 → 1-5
            
        macro_f1 = f1_score(labels, self.oof_preds, average="macro")
        print(f"\n{'='*50}")
        print(f"Overall OOF Macro F1: {macro_f1:.4f}")
        print(f"{'='*50}")
        return macro_f1
    
    #  Public: predict test set 
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Ensemble predict: average softmax prob từ tất cả fold models.
        Trả về nhãn 1-5.
        """
        texts = self.build_texts(df)
        test_ds = PaperDataset(texts, None, self.tokenizer, self.max_length)
        test_loader = DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
        )
        
        all_probs = np.zeros((len(texts), self.num_classes))
        for model in self.models:
            model.eval()
            model.to(DEVICE)
            fold_probs = []
            with torch.no_grad():
                for batch in test_loader:
                    input_ids      = batch["input_ids"].to(DEVICE)
                    attention_mask = batch["attention_mask"].to(DEVICE)
                    logits = model(input_ids, attention_mask)
                    probs  = torch.softmax(logits, dim=1).cpu().numpy()
                    fold_probs.append(probs)
            all_probs += np.vstack(fold_probs)
            
        all_probs /= len(self.models)
        return np.argmax(all_probs, axis=1) + 1  # 0-4 → 1-5
    
    #  Utility: save / load 
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