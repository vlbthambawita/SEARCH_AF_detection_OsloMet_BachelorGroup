import argparse
from pathlib import Path
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from loader import load_kfold, ECGDataset
from models.cnn1d import CNN1D
from models.resnet1d import ResNet1D


# ================= REPRODUCIBILITY =================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================= FIXED SETTINGS =================
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
KFOLDS = 5
EARLY_STOPPING_PATIENCE = 10
# ================================================


# ---------- Dataset statistics ----------
def print_dataset_stats(name, dataset):
    labels = [dataset[i][1] for i in range(len(dataset))]
    total = len(labels)
    afib = sum(1 for y in labels if y == 1)
    normal = total - afib

    print(
        f"{name:<12}: total={total} | "
        f"Normal={normal} ({100*normal/total:.2f}%) | "
        f"AFIB={afib} ({100*afib/total:.2f}%)"
    )


# ---------- Metrics ----------
def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return 2 * precision * recall / (precision + recall + 1e-12)


# ---------- Evaluation (VAL / TEST) ----------
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = total = correct = 0
    tp = tn = fp = fn = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).long()

            logits = model(x)
            loss = loss_fn(logits, y)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            correct += (preds == y).sum().item()

            tp += ((preds == 1) & (y == 1)).sum().item()
            tn += ((preds == 0) & (y == 0)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()

    acc = correct / total
    f1 = compute_f1(tp, fp, fn)
    cm = [[tn, fp], [fn, tp]]

    return acc, f1, cm, total_loss / total


# ---------- Train ONE fold ----------
def train_one_fold(model, optimizer, train_loader, val_loader, device, out_dir):
    loss_fn = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_f1 = -1
    best_acc = None
    best_cm = None
    bad_epochs = 0

    best_epoch = None
    best_val_loss = None


    out_dir.mkdir(parents=True, exist_ok=True)
    fold_start = time.time()

    print()  # space before epochs

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ---- Training ----
        model.train()
        train_loss = total = 0

        for x, y in train_loader:
            x = x.to(device).float()
            y = y.to(device).long()

            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y.size(0)
            total += y.size(0)

        train_loss /= total

        # ---- Validation ----
        acc, f1, cm, val_loss = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"TrainLoss {train_loss:.4f} | "
            f"ValLoss {val_loss:.4f} | "
            f"ACC {acc*100:.2f}% | "
            f"F1 {f1:.4f} | "
            f"Time {epoch_time:.2f}s"
        )

        torch.save(model.state_dict(), out_dir / "last.pt")

        # ---- Early stopping ----
        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_cm = cm
            best_epoch = epoch
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), out_dir / "best.pt")

        else:
            bad_epochs += 1

        if bad_epochs >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered")
            break

    fold_time = time.time() - fold_start

    tn, fp = best_cm[0]
    fn, tp = best_cm[1]

    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    precision = tp / (tp + fp + 1e-12)

    print("\n" + "=" * 70)
    print(f"Fold Results – Training time: {fold_time/60:.2f} minutes")
    print("Confusion Matrix (Validation)")
    print(f"[[{tn:4d} {fp:3d}]")
    print(f" [{fn:4d} {tp:3d}]]\n")

    print("Best F1    Accuracy   Recall(Sens)  Specificity  Precision")
    print("-" * 58)
    print(
        f"{best_f1:<10.4f}"
        f"{best_acc:<11.4f}"
        f"{recall:<15.4f}"
        f"{specificity:<13.4f}"
        f"{precision:<10.4f}"
    )
    print("=" * 70)
    
    metrics_path = out_dir / "metrics.txt"

    tn, fp = best_cm[0]
    fn, tp = best_cm[1]

    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    precision = tp / (tp + fp + 1e-12)

    with open(metrics_path, "w") as f:
        f.write("BEST VALIDATION METRICS (PER FOLD)\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Fold               : {out_dir.name}\n")
        f.write(f"Best epoch         : {best_epoch}\n")
        f.write(f"Validation loss    : {best_val_loss:.4f}\n")
        f.write(f"Validation F1      : {best_f1:.4f}\n")
        f.write(f"Validation Accuracy: {best_acc:.4f}\n\n")

        f.write("CONFUSION MATRIX (VAL)\n")
        f.write(f"[[{tn} {fp}]\n")
        f.write(f" [{fn} {tp}]]\n\n")

        f.write("DERIVED METRICS\n")
        f.write(f"Recall (Sensitivity): {recall:.4f}\n")
        f.write(f"Specificity        : {specificity:.4f}\n")
        f.write(f"Precision          : {precision:.4f}\n\n")

        f.write("RUN INFO\n")
        f.write("-" * 60 + "\n")
        f.write("Random seed        : 42\n")

    
    return best_f1


# ============================ MAIN ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model", choices=["cnn1d", "resnet1d"], default="cnn1d")
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = Path(args.data_path)
    fs = int(data_dir.name.replace("hz", ""))
    dataset_name = data_dir.parent.name

    print("=" * 70)
    print("ECG K-FOLD TRAINING")
    print("=" * 70)
    print(f"Device        : {device}")
    print(f"Dataset       : {dataset_name}")
    print(f"Sampling rate : {fs} Hz")
    print(f"Model         : {args.model}")
    print("=" * 70)

    best_fold = None
    best_f1_overall = -1

    training_start = time.time()


    # ---------- K-FOLD LOOP ----------
    for fold in range(1, KFOLDS + 1):
        print(f"\n=== Fold {fold}/{KFOLDS} ===")

        train_ds, val_ds, num_classes = load_kfold(data_dir, fs, fold, KFOLDS)

        print_dataset_stats("Train", train_ds)
        print_dataset_stats("Validation", val_ds)

        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, BATCH_SIZE)

        in_ch = train_ds[0][0].shape[0]
        model = (
            CNN1D(in_ch, num_classes)
            if args.model == "cnn1d"
            else ResNet1D(in_ch, num_classes)
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        fold_f1 = train_one_fold(
            model,
            optimizer,
            train_loader,
            val_loader,
            device,
            Path("checkpoints") / dataset_name / f"{fs}hz" / args.model / f"fold_{fold}",
        )

        if fold_f1 > best_f1_overall:
            best_f1_overall = fold_f1
            best_fold = fold


    training_total_time = time.time() - training_start
    training_total_min = training_total_time / 60
    print(f"Total training time : {training_total_min:.2f} minutes")


    # ================= FINAL TEST =================

    test_dir = data_dir / "test"
    if not test_dir.exists():
        print("\nTest folder not found — skipping test evaluation.")
        return

    print("\n" + "=" * 70)
    print(f"FINAL TEST EVALUATION (Best Fold = {best_fold})")
    print("=" * 70)

    test_data = torch.load(test_dir / "test.pt", map_location="cpu")
    test_ds = ECGDataset.__new__(ECGDataset)
    test_ds.X = test_data["X"]
    test_ds.y = test_data["y"]

    test_loader = DataLoader(test_ds, BATCH_SIZE)

    best_model_path = (
        Path("checkpoints") / dataset_name / f"{fs}hz" / args.model / f"fold_{best_fold}" / "best.pt"
    )

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    acc, f1, cm, _ = evaluate(model, test_loader, device)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start

    tn, fp = cm[0]
    fn, tp = cm[1]

    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    precision = tp / (tp + fp + 1e-12)

    # ================= SAVE FINAL TEST RESULTS =================


    results_dir = (
        Path("checkpoints")
        / dataset_name
        / f"{fs}hz"
        / args.model
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "test_results.txt"

    with open(results_path, "w") as f:
        # ---------- FINAL RESULTS ----------
        f.write(f"FINAL TEST RESULTS (Best Fold = {best_fold})\n")
        f.write("=" * 60 + "\n\n")

        f.write("CONFUSION MATRIX (TEST)\n")
        f.write("-" * 60 + "\n")
        f.write(f"[[{tn}  {fp}]\n")
        f.write(f" [{fn}   {tp}]]\n\n")

        f.write("METRICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy            : {acc:.4f}\n")
        f.write(f"F1-score            : {f1:.4f}\n")
        f.write(f"Recall (Sensitivity): {recall:.4f}\n")
        f.write(f"Specificity         : {specificity:.4f}\n")
        f.write(f"Precision           : {precision:.4f}\n\n")

        f.write("PERFORMANCE\n")
        f.write("-" * 60 + "\n")
        f.write(f"Training time (total) : {training_total_min:.2f} minutes\n")

        f.write(f"Inference time      : {elapsed:.2f} seconds\n")
        f.write(f"Throughput          : {len(test_ds)/elapsed:.2f} samples/sec\n\n\n")

        # ---------- RUN INFO ----------
        f.write("RUN INFORMATION\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset            : {dataset_name}\n")
        f.write(f"Sampling rate      : {fs} Hz\n")
        f.write(f"Model              : {args.model}\n")
        f.write(f"Best fold          : {best_fold}\n")
        f.write(f"Device             : {device}\n\n")

        f.write("DATASET SIZES\n")
        f.write("-" * 60 + "\n")
        f.write(f"Train samples      : {len(train_ds)}\n")
        f.write(f"Validation samples : {len(val_ds)}\n")
        f.write(f"Test samples       : {len(test_ds)}\n\n")

        f.write("TRAINING SETTINGS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Epochs (max)       : {EPOCHS}\n")
        f.write(f"Batch size         : {BATCH_SIZE}\n")
        f.write(f"Learning rate      : {LEARNING_RATE}\n")
        f.write(f"K-Folds            : {KFOLDS}\n")
        f.write(f"Early stop patience: {EARLY_STOPPING_PATIENCE}\n")



    # ---------- Save confusion matrix as CSV ----------
    cm_csv = results_dir / "confusion_matrix_test.csv"

    with open(cm_csv, "w") as f:
        f.write("TN,FP\n")
        f.write(f"{tn},{fp}\n")
        f.write("FN,TP\n")
        f.write(f"{fn},{tp}\n")



    print("Confusion Matrix (Test)")
    print(f"[[{tn:4d} {fp:3d}]")
    print(f" [{fn:4d} {tp:3d}]]\n")

    print("F1        Accuracy   Recall(Sens)  Specificity  Precision")
    print("-" * 58)
    print(
        f"{f1:<9.4f}"
        f"{acc:<11.4f}"
        f"{recall:<15.4f}"
        f"{specificity:<13.4f}"
        f"{precision:<10.4f}"
    )

    print(f"\nInference time : {elapsed:.2f} seconds")
    print(f"Throughput     : {len(test_ds)/elapsed:.2f} samples/sec")
    print("=" * 70)


if __name__ == "__main__":
    main()
