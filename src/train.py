'''PestAlertNet Phase 4 (L4): hybrid LSTM+Transformer outbreak training.

Dataset/artifact contract (Phase 2):
  - data/processed/sequences.npz
    keys: X [N,21,25] float32, y [N] {0,1}, split [N] {0:train,1:val,2:test}
    (and cell/date for audit only)
  - NOTE: All scaling was already done train-only in Phase 2; L4 does NOT refit scalers.

Model (Phase 3, unchanged):
  src.model.HybridOutbreakModel

Leakage protections (non-negotiable):
  - test split is only used in the final evaluation step
  - best checkpoint is selected on VALIDATION metrics only
  - no fitting/scaling with val/test data (there is no scaler refit here)
  - BCEWithLogitsLoss pos_weight computed from train split only

Approved L4 hyperparameters (documented in run_summary.json):
  - BATCH=64 (training), VAL_BATCH=256, MAX_EPOCHS=80
  - LR=1e-3, weight_decay=1e-4
  - ReduceLROnPlateau(mode="min", factor=0.5, patience=2, min_lr=1e-6, verbose=False)
  - early stopping: validation F1 primary, validation loss tie-breaker, patience=10
  - gradient clipping ON: clip_grad_norm(max_norm=1.0)
  - seed=42, threshold=0.5
  - device: CUDA if available else CPU
'''

import csv
import json
import math
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import HybridOutbreakModel


# ---------------------------------------------------------------- reproducibility
SEED = 42


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- device
def _device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------- dataset
def _make_datasets(data: Dict[str, np.ndarray]) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    X = torch.from_numpy(data['X']).float()       # [N,21,25]
    y = torch.from_numpy(data['y']).float()       # [N]
    split = data['split'].astype(np.int8)         # [N]

    train_ds = TensorDataset(X[split == 0], y[split == 0])
    val_ds = TensorDataset(X[split == 1], y[split == 1])
    test_ds = TensorDataset(X[split == 2], y[split == 2])
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------- metrics
def _metrics_from_probs(probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    preds = (probs >= threshold).astype(np.int8)
    tp = int(((preds == 1) & (targets == 1)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    auc = _binary_auc(targets, probs)

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
    }


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # ROC AUC via Mann-Whitney rank statistic; ties receive average ranks.
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_pos = int(y_true.sum())
    n_neg = int(y_true.size - n_pos)

    if n_pos == 0 or n_neg == 0:
        return float('nan')

    order = np.argsort(y_score, kind='mergesort')
    scores = y_score[order]
    labels = y_true[order].astype(np.float64)

    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and scores[end] == scores[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + 1 + end)
        start = end

    sum_positive_ranks = float((ranks * labels).sum())
    auc = (
        sum_positive_ranks - n_pos * (n_pos + 1) / 2.0
    ) / (n_pos * n_neg)

    return float(auc)

def _eval_epoch(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    '''Inference-only mini-batched evaluation. Returns loss, probs, targets (CPU).'''
    model.eval()
    total_loss = 0.0
    n = 0
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_targets.append(yb.detach().cpu().numpy())

    if n == 0:
        return {'loss': 0.0, 'probs': np.empty(0, dtype=np.float32), 'targets': np.empty(0, dtype=np.float32)}

    probs = np.concatenate(all_probs, axis=0).astype(np.float32)
    targets = np.concatenate(all_targets, axis=0).astype(np.float32)
    loss = total_loss / max(n, 1)
    metrics = _metrics_from_probs(probs, targets) if len(probs) else {}
    metrics['loss'] = loss
    return {'loss': loss, 'probs': probs, 'targets': targets, 'metrics': metrics}


# ---------------------------------------------------------------- checkpoint metadata
def _model_meta(model: HybridOutbreakModel) -> Dict[str, Any]:
    return {
        'input_dim': model.input_dim,
        'seq_len': model.seq_len,
        'hidden_size': model.hidden_size,
        'num_lstm_layers': model.lstm.num_layers,
        'lstm_bidirectional': model.lstm.bidirectional,
        'lstm_dropout': float(model.lstm.dropout) if model.lstm.dropout else 0.0,
        'd_model': model.ff_hidden,
        'num_heads': model.num_heads,
        'num_transformer_layers': model.transformer_encoder.num_layers,
        'dim_feedforward': model.transformer_encoder.layers[0].linear1.out_features,
        'transformer_dropout': float(model.transformer_encoder.layers[0].dropout.p),
        'mlp_hidden': model.classifier[0].out_features,
        'fused_dim': model.classifier[0].in_features,
        'classifier_dropout': float(model.classifier[2].p),
    }


def _save_best_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
                          val_f1: float, val_loss: float, pos_weight: float,
                          train_positive_count: int, train_negative_count: int, seed: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    payload: Dict[str, Any] = {
        'epoch': int(epoch),
        'best_val_f1': float(val_f1),
        'best_val_loss': float(val_loss),
        'seed': int(seed),
        'train_positive_count': int(train_positive_count),
        'train_negative_count': int(train_negative_count),
        'pos_weight': float(pos_weight),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': _model_meta(model),
    }
    torch.save(payload, path)


def _run_epoch_training(model: torch.nn.Module, train_loader: DataLoader, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer,
                        device: torch.device, clip_max_norm: float = 1.0) -> Dict[str, Any]:
    model.train()
    total_loss = 0.0
    n = 0
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
        optimizer.step()

        total_loss += float(loss.item()) * xb.size(0)
        n += xb.size(0)
        all_probs.append(torch.sigmoid(logits.detach()).cpu().numpy())
        all_targets.append(yb.detach().cpu().numpy())

    probs = np.concatenate(all_probs, axis=0).astype(np.float32)
    targets = np.concatenate(all_targets, axis=0).astype(np.float32)
    loss = total_loss / max(n, 1)
    metrics = _metrics_from_probs(probs, targets)
    metrics['loss'] = loss
    return metrics


def _log_lrs(optimizer: torch.optim.Optimizer) -> List[float]:
    return [group['lr'] for group in optimizer.param_groups]


# ---------------------------------------------------------------- main
def main() -> None:
    _set_seed(SEED)
    device = _device()
    start_time = time.time()

    npz = np.load('data/processed/sequences.npz')
    if not {'X', 'y', 'split'}.issubset(npz.files):
        raise KeyError('Expected keys X, y, split in data/processed/sequences.npz')

    train_ds, val_ds, test_ds = _make_datasets(npz)

    # train-only class stats / pos_weight
    train_y = torch.from_numpy(npz['y'][npz['split'] == 0]).float()
    train_positive_count = int((train_y == 1).sum().item())
    train_negative_count = int((train_y == 0).sum().item())
    p_train_pos = train_positive_count / max(train_positive_count + train_negative_count, 1)
    pos_weight = float((1 - p_train_pos) / max(p_train_pos, 1e-6))
    pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    BATCH = 64
    VAL_BATCH = 256
    MAX_EPOCHS = 80
    LR = 1e-3
    WD = 1e-4
    PATIENCE = 10

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=False,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=VAL_BATCH, shuffle=False)

    model = HybridOutbreakModel().to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)

    # early stopping state (validation metrics only)
    best_val_f1 = -float('inf')
    best_val_loss = float('inf')
    no_improve = 0
    best_epoch = -1
    best_state: Dict[str, Any] = {
        'f1': float('nan'),
        'loss': float('nan'),
        'epoch': -1,
        'lrs': _log_lrs(optimizer),
    }
    best_train_metrics: Dict[str, float] = {
        'loss': float('nan'),
        'f1': float('nan'),
        'precision': float('nan'),
        'recall': float('nan'),
        'auc': float('nan'),
    }

    epoch_rows: List[Dict[str, Any]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        tr = _run_epoch_training(model, train_loader, criterion, optimizer, device)
        ev = _eval_epoch(model, val_loader, criterion, device)
        val_loss = ev['loss']
        val_f1 = ev['metrics'].get('f1', float('nan'))

        # scheduler sees validation loss only
        scheduler.step(val_loss)

        epoch_rows.append({
            'epoch': int(epoch),
            'train_loss': float(tr['loss']),
            'val_loss': float(val_loss),
            'train_f1': float(tr.get('f1', float('nan'))),
            'val_f1': float(val_f1),
            'train_precision': float(tr.get('precision', float('nan'))),
            'val_precision': float(ev['metrics'].get('precision', float('nan'))),
            'train_recall': float(tr.get('recall', float('nan'))),
            'val_recall': float(ev['metrics'].get('recall', float('nan'))),
            'train_auc': float(tr.get('auc', float('nan'))),
            'val_auc': float(ev['metrics'].get('auc', float('nan'))),
            'lr': float(optimizer.param_groups[0]['lr']),
        })

        improved = False
        if math.isfinite(val_f1) and (val_f1 > best_val_f1):
            improved = True
        elif math.isfinite(val_f1) and (val_f1 == best_val_f1) and (val_loss < best_val_loss):
            improved = True

        if improved:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            best_epoch = epoch
            best_train_metrics = {
                'loss': float(tr['loss']),
                'f1': float(tr.get('f1', float('nan'))),
                'precision': float(tr.get('precision', float('nan'))),
                'recall': float(tr.get('recall', float('nan'))),
                'auc': float(tr.get('auc', float('nan'))),
            }
            no_improve = 0
            _save_best_checkpoint('models/best_model.pt', model, optimizer, epoch,
                                 best_val_f1, best_val_loss, pos_weight,
                                 train_positive_count, train_negative_count, SEED)
            best_state = {
                'f1': best_val_f1,
                'loss': best_val_loss,
                'epoch': best_epoch,
                'lrs': _log_lrs(optimizer),
            }
        else:
            no_improve += 1

        # optional per-epoch metrics file
        with open('outputs/l4_epoch_metrics.csv', 'w', newline='', encoding='utf-8') as fh:
            if epoch_rows:
                writer = csv.DictWriter(fh, fieldnames=list(epoch_rows[0].keys()))
                writer.writeheader()
                for row in epoch_rows:
                    writer.writerow(row)

        if no_improve >= PATIENCE:
            # final fallback: if no improvement ever observed, save last model
            if best_epoch == -1:
                _save_best_checkpoint('models/best_model.pt', model, optimizer, epoch,
                                     float('nan'), float('nan'), pos_weight,
                                     train_positive_count, train_negative_count, SEED)
            break

    # ---- test evaluation AFTER training + best-checkpoint selection ----
    # load best checkpoint deterministically
    ckpt = torch.load('models/best_model.pt', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    te = _eval_epoch(model, test_loader, criterion, device)

    # ---- run summary ----
    summary = {
        'seed': SEED,
        'device': str(device),
        'batch_size_training': BATCH,
        'batch_size_validation': VAL_BATCH,
        'max_epochs': MAX_EPOCHS,
        'lr': LR,
        'weight_decay': WD,
        'scheduler': {
            'type': 'ReduceLROnPlateau',
            'mode': 'min',
            'factor': 0.5,
            'patience': 2,
            'min_lr': 1e-6,
            'verbose': False,
            'monitor': 'val_loss',
        },
        'early_stopping': {
            'primary': 'val_f1',
            'tie_break': 'val_loss',
            'patience': PATIENCE,
        },
        'gradient_clipping': {
            'enabled': True,
            'max_norm': 1.0,
            'applied_after_backward_before_optimizer_step': True,
        },
        'threshold': 0.5,
        'class_weighting': {
            'method': 'BCEWithLogitsLoss pos_weight computed from train split only',
            'pos_weight': pos_weight,
            'train_positive_count': train_positive_count,
            'train_negative_count': train_negative_count,
        },
        'train_epochs_completed': int(epoch),
        'best_checkpoint': {
            'epoch': best_state['epoch'],
            'val_f1': best_state['f1'],
            'val_loss': best_state['loss'],
            'path': 'models/best_model.pt',
        },
        'train_performance_at_best': {
            'loss': float(best_train_metrics['loss']),
            'f1': float(best_train_metrics['f1']),
            'precision': float(best_train_metrics['precision']),
            'recall': float(best_train_metrics['recall']),
            'auc': float(best_train_metrics['auc']),
        },
        'validation_performance_at_best': {
            'loss': float(best_val_loss) if best_state['epoch'] != -1 else float('nan'),
            'f1': float(best_val_f1) if best_state['epoch'] != -1 else float('nan'),
        },
        'test_performance': {
            'loss': float(te['loss']),
            'f1': float(te['metrics'].get('f1', float('nan'))),
            'precision': float(te['metrics'].get('precision', float('nan'))),
            'recall': float(te['metrics'].get('recall', float('nan'))),
            'auc': float(te['metrics'].get('auc', float('nan'))),
            'conf_matrix': {
                'tp': int(te['metrics'].get('tp', 0)),
                'fp': int(te['metrics'].get('fp', 0)),
                'fn': int(te['metrics'].get('fn', 0)),
                'tn': int(te['metrics'].get('tn', 0)),
            },
        },
        'model_config': _model_meta(model),
        'artifact_loaded': 'data/processed/sequences.npz',
        'splits_used': {'train': int(train_ds.__len__()), 'val': int(val_ds.__len__()), 'test': int(test_ds.__len__())},
        'elapsed_seconds': float(time.time() - start_time),
    }
    os.makedirs(os.path.dirname('outputs/l4_run_summary.json'), exist_ok=True)
    with open('outputs/l4_run_summary.json', 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=1)

    print('L4 training complete')
    print(f'  best_checkpoint epoch={best_state["epoch"]} val_f1={best_state["f1"]} val_loss={best_state["loss"]}')
    print(f'  test_f1={summary["test_performance"]["f1"]} test_auc={summary["test_performance"]["auc"]}')


if __name__ == '__main__':
    main()
