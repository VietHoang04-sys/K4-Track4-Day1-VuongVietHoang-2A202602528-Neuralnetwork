"""Load the fixed CoverType split, create validation data, and standardize features.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    from pathlib import Path

    root = Path(processed_dir)
    train_path, eval_path = root / "train.npz", root / "eval.npz"
    if not train_path.is_file() or not eval_path.is_file():
        raise FileNotFoundError(
            f"Missing processed data under {root}. Run `python scripts/split_data.py` from the repository root."
        )
    with np.load(train_path) as train, np.load(eval_path) as evaluation:
        X_train = train["X"]
        y_train = train["y"]
        X_eval = evaluation["X"]
        y_eval = evaluation["y"]
        eval_row_id = evaluation["row_id"]

    if X_train.ndim != 2 or X_train.shape[1] != 54 or X_eval.ndim != 2 or X_eval.shape[1] != 54:
        raise ValueError("Expected train and eval feature arrays with shape (N, 54).")
    if y_train.dtype != np.int64 or y_eval.dtype != np.int64:
        raise TypeError("Labels must have dtype int64.")
    if X_train.dtype != np.float32 or X_eval.dtype != np.float32:
        raise TypeError("Features must have dtype float32.")
    if len(X_train) != len(y_train) or len(X_eval) != len(y_eval) or len(X_eval) != len(eval_row_id):
        raise ValueError("Feature, label, and row_id lengths do not match.")
    if not np.isin(y_train, np.arange(7)).all() or not np.isin(y_eval, np.arange(7)).all():
        raise ValueError("Labels must be integers in the range 0..6.")
    return X_train, y_train, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    from sklearn.model_selection import train_test_split

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1.")
    return train_test_split(
        X, y, test_size=val_fraction, random_state=seed, stratify=y
    )


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    X_tr = np.asarray(X_tr)
    if X_tr.ndim != 2 or X_tr.shape[1] < N_NUMERIC:
        raise ValueError(f"X_tr must have at least {N_NUMERIC} feature columns.")
    mean = X_tr[:, :N_NUMERIC].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X_tr[:, :N_NUMERIC].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std == 0] = 1.0
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] < N_NUMERIC:
        raise ValueError(f"X must have at least {N_NUMERIC} feature columns.")
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (N_NUMERIC,) or std.shape != (N_NUMERIC,):
        raise ValueError(f"mean and std must each have shape ({N_NUMERIC},).")
    safe_std = np.where(std == 0, 1.0, std)
    result = X.astype(np.float32, copy=True)
    result[:, :N_NUMERIC] = (result[:, :N_NUMERIC] - mean) / safe_std
    return result


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, X_val, y_tr, y_val = make_val_split(
        X_train_full, y_train_full, val_fraction=val_fraction, seed=seed
    )
    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    resolved_device = torch.device(device)
    tensors = {
        "X_tr": torch.as_tensor(X_tr, dtype=torch.float32, device=resolved_device),
        "y_tr": torch.as_tensor(y_tr, dtype=torch.int64, device=resolved_device),
        "X_val": torch.as_tensor(X_val, dtype=torch.float32, device=resolved_device),
        "y_val": torch.as_tensor(y_val, dtype=torch.int64, device=resolved_device),
        "X_eval": torch.as_tensor(X_eval, dtype=torch.float32, device=resolved_device),
        "y_eval": torch.as_tensor(y_eval, dtype=torch.int64, device=resolved_device),
    }
    majority = int(np.bincount(y_val, minlength=7).argmax())
    majority_accuracy = float(np.mean(y_val == majority))
    print(
        f"Train: {len(y_tr):,}; val: {len(y_val):,}; eval: {len(y_eval):,}; "
        f"device: {resolved_device}; majority-class val accuracy: {majority_accuracy:.4f}"
    )
    print(
        "Standardized numeric feature mean/std (train only): "
        f"{X_tr[:, :N_NUMERIC].mean(axis=0).round(4)} / "
        f"{X_tr[:, :N_NUMERIC].std(axis=0).round(4)}"
    )
    tensors.update(
        eval_row_id=np.asarray(eval_row_id, dtype=np.int64),
        standardizer_mean=mean,
        standardizer_std=std,
        majority_class=majority,
        majority_val_accuracy=majority_accuracy,
    )
    return tensors


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    n_samples = len(X)
    if len(y) != n_samples:
        raise ValueError("X and y must contain the same number of samples.")
    if shuffle:
        perm = torch.randperm(
            n_samples, generator=generator, device=X.device
        )
    else:
        perm = torch.arange(n_samples, device=X.device)
    for start in range(0, n_samples, batch_size):
        indices = perm[start : start + batch_size]
        yield X[indices], y[indices]
