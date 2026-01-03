"""
ecg_data_preprocessor.py

Universal ECG preprocessing 
----------------------------------
This script serves as the global standard for ECG signal processing in this pipeline. 
It handles resampling, normalization, segmentation, and patient-safe balancing.
This logic is finalized and should not be modified unless implementing global feature updates 
for example, lead selection or multi-label support.

what this script/pipeline do:
- Convert raw ECG records into clean, fixed-length segments(10s) if they are not already.
- Resample to multiple target sampling rates for example 500/250/100 Hz or any other rates.
    (the pipeline does NOT resample when the original sampling rate already matches the target).
- Enforce patient-safe splitting (train/val/test or K-fold)
- Apply record-level AFIB vs NORMAL balancing
- Save processed tensors (.pt) and metadata (.csv)

Leakage safaty garantee
-------------------------------------------------------------------------------
- Patients are the main unit of splitting.
- No patient ever appears in more than one fold or split.
- Segments inherit the patient assignment of their parent record.
- PTB-XL official strat_fold is respected when folds=10.

"""

import os
import csv
import logging
from dataclasses import dataclass
from typing import List
from collections import defaultdict, Counter

import numpy as np
import torch
from scipy.signal import resample
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm.auto import tqdm


# ================= Data Structure =================

@dataclass
# Represent a single ECG record and its associated metadata.
class Record:               
    signal: np.ndarray
    fs: int
    label: int
    patient_id: str
    record_id: str
    fold: int | None = None



# ================= Logger Setup =================

def setup_logger(log_path: str) -> logging.Logger:
    """
    Creates a logger that writes both to console and to a file.

    Ensures:
        - Log directory exists
        - Log file is overwritten each run
        - Consistent formatting
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

    # File handler
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Consol handler
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info("Logger initialized successfully")
    return logger
    

# ================= Fold Utilities =================

def assert_no_patient_leakage(folds_dict):
    """
    Make sure that no patient appears in more than one fold.

    """
    seen = set()
    for fold, pids in folds_dict.items():
        overlap = seen & pids
        if overlap:
            raise RuntimeError(f"Patient leakage detected in fold {fold}: {overlap}")
        seen |= pids


def build_patient_folds(records: List[Record], k: int, seed=42):
    """
    Builds patient-safe stratified folds.
        - If PTB-XL official folds exist AND k=10, then use official folds.
        - Otherwise, compute patient-level labels and run StratifiedKFold.

    returns: dict: fold_id -> set(patient_ids)
    """

    # PTB‑XL official fold mode
    if records and records[0].fold is not None and k == 10:
        folds = defaultdict(set)
        for r in records:
            folds[r.fold].add(r.patient_id)
        assert_no_patient_leakage(folds)
        return dict(folds)

    # Build patient‑level labels
    patient_labels = defaultdict(list)
    for r in records:
        patient_labels[r.patient_id].append(r.label)

    patients = list(patient_labels.keys())
    labels = [int(np.round(np.mean(patient_labels[p]))) for p in patients]

    # Stratified K‑fold on patient level
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = {}

    for i, (_, idx) in enumerate(skf.split(patients, labels), start=1):
        folds[i] = {patients[j] for j in idx}

    assert_no_patient_leakage(folds)
    return folds


# ================= Signal Processing =================

def clean_signal(x):
    """
    Replace NaN and Inf values with zeros to avoid numerical issues.
    """
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def resample_signal(x, fs_in, fs_out):
    """
    Resample signal from fs_in to fs_out using scipy.signal.resample.

    If sampling rate is unchanged, returns the input as float32.
    """
    if fs_in == fs_out:
        return x.astype(np.float32)
    n = int(round(len(x) * fs_out / fs_in))
    return resample(x, n).astype(np.float32)


def zscore(x):
    """
    Apply z-score normalization per lead (channel).
    """
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-8
    return ((x - mean) / std).astype(np.float32)


def make_segments(x, seg_len):
    """
    Slice signal into non-overlapping fixed-length segments.

    Returns: List of segments, each of length seg_len.
    """
    return [x[i:i + seg_len] for i in range(0, len(x) - seg_len + 1, seg_len)]

# ================= Signal Quality Corrections =================
#  Flatline leads  are zeroed per segment
#  Extreme amplitude outliers are clipped to stabilize training


def zero_flatline_leads(x, eps=1e-6, min_flat_fraction=0.5):
    """
    Zero leads that are flat for too long.
    x: (C, T)
    returns: (x_fixed, num_zeroed_leads)
    """
    C, T = x.shape
    zeroed = 0

    for c in range(C):
        dx = np.abs(np.diff(x[c]))

        signal_scale = np.std(x[c]) + 1e-8
        flat_fraction = np.mean(dx < (eps * signal_scale))

        if flat_fraction >= min_flat_fraction:
            x[c] = 0.0
            zeroed += 1

    return x, zeroed


def clip_extremes(x, clip_value=15.0):
    """
    Clip extreme amplitudes.
    returns: (x_clipped, num_clipped_values)
    """
    before = np.abs(x) > clip_value
    n_clipped = int(before.sum())
    x = np.clip(x, -clip_value, clip_value)
    return x, n_clipped



# ================= Hold-out Test Split =================

def split_patients_test(records: List[Record], test_ratio: float, seed=42):
    """
    Patient-safe hold-out test split.
    Returns:
        trainval_records, test_records
    """
    pid_to_labels = defaultdict(list)
    for r in records:
        pid_to_labels[r.patient_id].append(r.label)

    patients = list(pid_to_labels.keys())
    labels = [int(np.round(np.mean(pid_to_labels[p]))) for p in patients]

    trainval_pids, test_pids = train_test_split(
        patients,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
    )

    trainval_pids = set(trainval_pids)
    test_pids = set(test_pids)

    trainval_records = [r for r in records if r.patient_id in trainval_pids]
    test_records     = [r for r in records if r.patient_id in test_pids]

    return trainval_records, test_records


# ================= Main Pipeline =================

def prepare_dataset(
    dataset_name: str,
    records: List[Record],
    out_root="prepared_data",
    target_fs=(500, 250, 100),
    segment_seconds=10,
    max_samples=None,
    split_ratio=(0.7, 0.2, 0.1),
    balance_mode="global",
    folds=None,
    test_ratio=None,
    seed=42,
):
    """
    Main preprocessing pipeline.

      This function takes raw Record objects and produces a fully processed,
    leakage-safe dataset ready for model training. It handles cleaning,
    resampling, normalization, segmentation, balancing, and patient-level
    splitting, and saves both metadata and tensor files to disk.
    """
    logger = setup_logger(os.path.join("logs", f"{dataset_name}.log"))


   # ================= Dataset Summary =================

    labels = [r.label for r in records]
    patients = {r.patient_id for r in records}
    c = Counter(labels)

    logger.info("DATASET SUMMARY (RAW RECORDS)")
    logger.info(f"  Total records  : {len(records)}")
    logger.info(f"  Total patients : {len(patients)}")
    logger.info(f"  AFIB records   : {c.get(1,0)}")
    logger.info(f"  NORMAL records : {c.get(0,0)}")



    # ================= Optional Hold-out Test Split (ADDED FEATURE) =================

    test_records = None

    if test_ratio is not None:
        logger.info(f"HOLD-OUT TEST MODE ENABLED (test_ratio={test_ratio})")

        records, test_records = split_patients_test(
            records, test_ratio, seed
        )

        logger.info(
            f"Patients after split: "
            f"train+val={len({r.patient_id for r in records})}, "
            f"test={len({r.patient_id for r in test_records})}"
        )



   # ================= Record-Level Balancing =================

    if balance_mode == "global" and folds is None:
        # Separate AFIB and NORMAL records
        afib = [r for r in records if r.label == 1]
        normal = [r for r in records if r.label == 0]

        # Balance by taking equal number of each
        n = min(len(afib), len(normal))
      

        rng = np.random.default_rng(seed)

        # Randomly sample balanced records
        records = list(rng.choice(afib, n, replace=False)) + \
                  list(rng.choice(normal, n, replace=False))
        rng.shuffle(records)

        logger.info("BALANCING RULE APPLIED (RECORD LEVEL)")
        logger.info(f"  AFIB kept   : {n}")
        logger.info(f"  NORMAL kept : {n}")
        logger.info(f"  Total kept  : {2*n}")


    # ================= Patient-Safe Splitting =================

    if folds is not None:
        # K‑fold mode
        logger.info(f"FOLD MODE ENABLED (K={folds})")
        patient_folds = build_patient_folds(records, folds, seed)
    else:
        # Standard train/val/test split (patient‑safe)
        pid_to_label = {r.patient_id: r.label for r in records}

        pids = np.array(list(pid_to_label.keys()))
        pid_labels = np.array([pid_to_label[p] for p in pids])

        # Train vs temp
        train_p, temp, train_y, temp_y = train_test_split(
            pids,
            pid_labels,
            test_size=1 - split_ratio[0],
            stratify=pid_labels,
            random_state=seed,
        )

        val_p, test_p, val_y, test_y = train_test_split(
            temp,
            temp_y,
            test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]),
            stratify=temp_y,
            random_state=seed,
        )


    # ================= ENSURE ≥1 AFIB PATIENT IN TRAIN (ONLY IF max_samples SET) =================

    if max_samples is not None and folds is None:
        train_afib_pids = {
            r.patient_id for r in records
            if r.patient_id in train_p and r.label == 1
        }

        if len(train_afib_pids) == 0:
            raise RuntimeError(
                "Invalid split with max_samples enabled: "
                "TRAIN contains no AFIB patients. "
                "Reduce test_ratio, increase max_samples, or change seed."
            )





    # ================= Process each Sampling Rate =================

    for fs in target_fs:
        out_dir = os.path.join(out_root, dataset_name, f"{fs}hz")
        os.makedirs(out_dir, exist_ok=True)

        seg_len = fs * segment_seconds
        segments = []

        records_with_clipping = 0
        records_with_flatlines = 0


        
        # Segment creation
        
        for r in tqdm(records, desc=f"{fs}Hz"):
            # Clean → resample → transpose → zscore → transpose back
            sig = resample_signal(clean_signal(r.signal), r.fs, fs).T

            sig, n_clipped = clip_extremes(sig, clip_value=15.0)
            sig, n_zeroed = zero_flatline_leads(sig)

            if n_clipped > 0:
                records_with_clipping += 1

            if n_zeroed > 0:
                records_with_flatlines += 1

            sig = zscore(sig)


            # Create fixed‑length segments
            for seg in make_segments(sig.T, seg_len):
                segments.append((seg, r.label, r.record_id, r.patient_id))

        logger.info(f"[{fs}Hz] SEGMENTS CREATED: {len(segments)}")

        if dataset_name.startswith(("ptbxl", "ecg-arrhythmia")):
            logger.info(
                "Note: For this dataset, one segment corresponds to one ECG record."
            )

        logger.info(
            f"[{fs}Hz] QC SUMMARY: "
            f"{records_with_clipping}/{len(records)} records had extreme-value clipping, "
            f"{records_with_flatlines}/{len(records)} records had at least 1 flatline lead"
        )




        # Segment distribution BEFORE balancing
        seg_labels = [s[1] for s in segments]
        seg_counter = Counter(seg_labels)

        logger.info(f"[{fs}Hz] SEGMENT DISTRIBUTION (BEFORE BALANCING)")
        logger.info(f"  AFIB segments   : {seg_counter.get(1, 0)}")
        logger.info(f"  NORMAL segments : {seg_counter.get(0, 0)}")
        logger.info(f"  TOTAL segments  : {len(segments)}")


        # ================= Global Segment Balancing =================


        #  If No Folds(not selected) so Global Segment Balancing 
        if folds is not None:
            afib = [s for s in segments if s[1] == 1]
            normal = [s for s in segments if s[1] == 0]

            rng = np.random.default_rng(seed)
            n = min(len(afib), len(normal))

            if max_samples is not None:
                n = min(n, max_samples // 2)

            if n == 0:
                raise RuntimeError("No segments available after balancing")

            # Randomly sample balanced segments
            afib_idx = rng.choice(len(afib), n, replace=False)
            norm_idx = rng.choice(len(normal), n, replace=False)

            segments = (
                [afib[i] for i in afib_idx] +
                [normal[i] for i in norm_idx]
            )
            rng.shuffle(segments)

            logger.info(f"[{fs}Hz] GLOBAL SEGMENT BALANCE APPLIED")
            logger.info(f"  AFIB segments   : {n}")
            logger.info(f"  NORMAL segments : {n}")
            logger.info(f"  TOTAL segments  : {2*n}")

        # ================= Distribute Segments =================

        splits = defaultdict(lambda: ([], [], [], []))


        

        for seg, lbl, rid, pid in segments:
            if folds is None:
                # All segments from the same patient go to the same split.
                split = (
                    "train" if pid in train_p else
                    "val" if pid in val_p else
                    "test"
                )
            else:
                # Finds which fold contains that patient, and assigns the segment to that fold.
                for fid, pset in patient_folds.items():
                    if pid in pset:
                        split = fid
                        break

            # Append segment to appropriate split/fold
            splits[split][0].append(seg)
            splits[split][1].append(lbl)
            splits[split][2].append(rid)
            splits[split][3].append(pid)

        

        # ================= GLOBAL SAMPLE CAP (ALL MODES) =================
        # Applied ONLY if max_samples is set

        if max_samples is not None:

            logger.info(f"[{fs}Hz] GLOBAL SAMPLE CAP APPLIED (max_samples={max_samples})")

            logger.info(
                f"[{fs}Hz] NOTE: max_samples is a TARGET, not a guarantee."
            )
            logger.info(
                f"[{fs}Hz] Reason: patient-level splitting and per-fold balancing "
                f"may require dropping samples to preserve patient integrity "
                f"and class balance."
            )


            # Identify CV vs TEST containers
            if folds is not None:
                cv_keys = list(splits.keys())   # fold ids
                test_key = "test"
            else:
                cv_keys = ["train", "val"]
                test_key = "test"

            # ---- budgets ----
            if test_ratio is not None:
                test_cap = int(test_ratio * max_samples)
            elif folds is None:
                # split mode: behave exactly like before (70/20/10)
                test_cap = int(split_ratio[2] * max_samples)
            else:
                test_cap = 0

            cv_cap   = max_samples - test_cap

            if folds is not None and test_ratio is not None:
                logger.info(
                    f"[{fs}Hz] CV cap (folds combined) : {cv_cap}"
                )
                logger.info(
                    f"[{fs}Hz] Test cap               : {test_cap}"
                )


            def cap_container(container, cap):
                X, y, rids, pids = container
                if len(y) <= cap:
                    return container

                rng = np.random.default_rng(seed)
                idx = rng.choice(len(y), cap, replace=False)

                return (
                    [X[i] for i in idx],
                    [y[i] for i in idx],
                    [rids[i] for i in idx],
                    [pids[i] for i in idx],
                )

            # ---- cap TEST ----
            if test_key in splits and test_cap > 0:
                splits[test_key] = cap_container(splits[test_key], test_cap)

            # ---- cap CV ----
            # flatten CV first
            cv_X, cv_y, cv_rids, cv_pids = [], [], [], []
            for k in cv_keys:
                X, y, rids, pids = splits[k]
                cv_X.extend(X)
                cv_y.extend(y)
                cv_rids.extend(rids)
                cv_pids.extend(pids)

            cv_capped = cap_container((cv_X, cv_y, cv_rids, cv_pids), cv_cap)

            # wipe old CV containers
            for k in cv_keys:
                splits[k] = ([], [], [], [])

            # reassign CV samples (folds handled later)
            splits["_cv_buffer"] = cv_capped

            # ================= APPLY CAPPED CV BUFFER =================
            # (ADDED — REQUIRED)

            if folds is not None:
                # Rebuild folds from capped CV buffer
                X, y, rids, pids = splits["_cv_buffer"]

                for fid in cv_keys:
                    splits[fid] = ([], [], [], [])

                for seg, lbl, rid, pid in zip(X, y, rids, pids):
                    for fid, pset in patient_folds.items():
                        if pid in pset:
                            splits[fid][0].append(seg)
                            splits[fid][1].append(lbl)
                            splits[fid][2].append(rid)
                            splits[fid][3].append(pid)
                            break

            else:
                # Classic split: rebuild train/val from capped CV
                X, y, rids, pids = splits["_cv_buffer"]

                splits["train"] = ([], [], [], [])
                splits["val"]   = ([], [], [], [])

                for seg, lbl, rid, pid in zip(X, y, rids, pids):
                    split = "train" if pid in train_p else "val"
                    splits[split][0].append(seg)
                    splits[split][1].append(lbl)
                    splits[split][2].append(rid)
                    splits[split][3].append(pid)

            del splits["_cv_buffer"]





        # ================= Train-Only Segment Balancing (Medical-Safe) =================


        if folds is None and balance_mode == "train":

            X, y, rids, pids = splits["train"]

            afib_idx = [i for i, lbl in enumerate(y) if lbl == 1]
            norm_idx = [i for i, lbl in enumerate(y) if lbl == 0]

            logger.info(
                f"[{fs}Hz] Train class counts before balancing: "
                f"AFIB={len(afib_idx)}, NORMAL={len(norm_idx)}"
            )
            n = min(len(afib_idx), len(norm_idx))

        

            if n == 0:
                logger.warning(
                    f"[{fs}Hz] Train balancing skipped: one class missing in train split "
                    f"(AFIB={len(afib_idx)}, NORMAL={len(norm_idx)})."
                )
                # Do NOT rebalance, keep train as-is
                pass
            else:
                rng = np.random.default_rng(seed)
                keep_idx = (
                    list(rng.choice(afib_idx, n, replace=False)) +
                    list(rng.choice(norm_idx, n, replace=False))
                )
                rng.shuffle(keep_idx)

                splits["train"] = (
                    [X[i] for i in keep_idx],
                    [y[i] for i in keep_idx],
                    [rids[i] for i in keep_idx],
                    [pids[i] for i in keep_idx],
                )

                logger.info(f"[{fs}Hz] TRAIN-ONLY SEGMENT BALANCE APPLIED")
                logger.info(f"  AFIB train segments   : {n}")
                logger.info(f"  NORMAL train segments : {n}")
                logger.info(f"  TOTAL train segments  : {2*n}")


        # ================= Fold-Level Segment Balancing =================

        if folds is not None:

            
            

            logger.info(f"[{fs}Hz] SEGMENT BALANCING APPLIED (FOLD LEVEL)")
            logger.info("  Rule: min(AFIB, NORMAL) per fold")

            per_fold_total = (
                max_samples // folds if max_samples is not None else None
            )

            #  accumulators (logging only)
            fold_logs = []        # store fold messages
            total_kept = 0
            kept_afib = 0
            kept_normal = 0

            for fid in list(splits.keys()):
                X, y, rids, pids = splits[fid]

                afib_idx = [i for i, lbl in enumerate(y) if lbl == 1]
                norm_idx = [i for i, lbl in enumerate(y) if lbl == 0]

                n = min(len(afib_idx), len(norm_idx))

                if per_fold_total is not None:
                    n = min(n, per_fold_total // 2)

                if n == 0:
                    logger.warning(f"[{fs}Hz] fold {fid} has no balanced segments")
                    splits[fid] = ([], [], [], [])
                    fold_logs.append((fid, 0, 0, 0))  # >>> ADDED
                    continue

                rng = np.random.default_rng(seed + int(fid))
                keep_idx = (
                    list(rng.choice(afib_idx, n, replace=False)) +
                    list(rng.choice(norm_idx, n, replace=False))
                )
                rng.shuffle(keep_idx)

                splits[fid] = (
                    [X[i] for i in keep_idx],
                    [y[i] for i in keep_idx],
                    [rids[i] for i in keep_idx],
                    [pids[i] for i in keep_idx],
                )

                #  collect stats only
                total_kept += 2 * n
                kept_afib += n
                kept_normal += n
                fold_logs.append((fid, 2*n, n, n))

            # GLOBAL summary (ONCE, before fold logs)
            logger.info(f"[{fs}Hz] SEGMENT BALANCING APPLIED")
            logger.info(f"  AFIB kept   : {kept_afib}")
            logger.info(f"  NORMAL kept : {kept_normal}")
            logger.info(f"  TOTAL kept  : {total_kept}")
            logger.info(f"  DROPPED     : {len(segments) - total_kept}")

        
            for fid, tot, a, n in fold_logs:
                logger.info(
                    f"[{fs}Hz] fold {fid}: segments={tot} (AFIB={a}, NORMAL={n})"
                )

                

        # ================= Save Data =================

        # Always save CSV metadata
        all_X, all_y, all_rids, all_pids, all_splits = [], [], [], [], []

        for split, (X, y, rids, pids) in splits.items():
            all_X.extend(X)
            all_y.extend(y)
            all_rids.extend(rids)
            all_pids.extend(pids)
            all_splits.extend([split] * len(y))

        
        all_pids  = [str(pid) for pid in all_pids]
        all_rids  = [str(rid) for rid in all_rids]
        all_y     = [int(lbl) for lbl in all_y]

        split_col = "fold" if folds is not None else "split"
        csv_path = os.path.join(out_dir, f"samples_{fs}hz.csv")
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [split_col, "patient_id", "record_id", "label", "fs", "segment_index"]
            )
            for i in range(len(all_y)):
                writer.writerow(
                    [all_splits[i], all_pids[i], all_rids[i], all_y[i], fs, i]
                )

        # Reshape if not correcect (N, C, T)

        def fix_shape(X_list):
            X_np = np.stack(X_list)  # (N, T, C) or (N, C, T)
            if X_np.ndim == 3 and X_np.shape[1] > X_np.shape[2]:
                X_np = np.transpose(X_np, (0, 2, 1))  # -> (N, C, T)
            return torch.tensor(X_np).contiguous()
        
        # Save data.pt only if Fold Mode
        if folds is not None:
            X_tensor = fix_shape(all_X)
            torch.save(
                {
                    "X": X_tensor,
                    "y": torch.tensor(all_y),
                    "record_ids": all_rids,
                    "patient_ids": all_pids,
                },
                os.path.join(out_dir, "data.pt"),
            )

            logger.info(f"[{fs}Hz] saved data.pt (fold mode)")

        # Save train.pt, val.pt, test.pt only if None Fold Mode
        else:
            for split in ("train", "val", "test"):
                X, y, rids, pids = splits.get(split, ([], [], [], []))

                if len(X) == 0:
                    logger.warning(f"[{fs}Hz] {split}.pt is empty — skipping")
                    continue

                X_tensor = fix_shape(X)

                torch.save(
                    {
                        "X": X_tensor,
                        "y": torch.tensor(y),
                        "record_ids": rids,
                        "patient_ids": pids,
                    },
                    os.path.join(out_dir, f"{split}.pt"),
                )

                logger.info(
                    f"[{fs}Hz] saved {split}.pt "
                    f"(samples={len(y)}, AFIB={sum(y)}, NORMAL={len(y)-sum(y)})"
                )

        # ================= Save HOLD-OUT TEST SET  =================

        if test_records is not None:
            test_dir = os.path.join(out_root, dataset_name, f"{fs}hz", "test")
            os.makedirs(test_dir, exist_ok=True)

            test_segments = []
            

            for r in test_records:
                sig = resample_signal(clean_signal(r.signal), r.fs, fs).T
                sig, _ = clip_extremes(sig)
                sig, _ = zero_flatline_leads(sig)
                sig = zscore(sig)

                for seg in make_segments(sig.T, seg_len):
                    test_segments.append((seg, r.label, r.record_id, r.patient_id))

            if max_samples is not None and test_ratio is not None:
                test_cap = int(test_ratio * max_samples)
                if len(test_segments) > test_cap:
                    rng = np.random.default_rng(seed)
                    idx = rng.choice(len(test_segments), test_cap, replace=False)
                    test_segments = [test_segments[i] for i in idx]

            if test_segments:
                X_t, y_t, rids, pids = zip(*test_segments)

                X_tensor = fix_shape(list(X_t))

                torch.save(
                    {
                        "X": X_tensor,
                        "y": torch.tensor(y_t),
                        "record_ids": list(rids),
                        "patient_ids": list(pids),
                    },
                    os.path.join(test_dir, "test.pt"),
                )

                logger.info(
                    f"[{fs}Hz] saved HOLD-OUT test.pt "
                    f"(samples={len(y_t)}, AFIB={sum(y_t)}, NORMAL={len(y_t)-sum(y_t)})"
                )

