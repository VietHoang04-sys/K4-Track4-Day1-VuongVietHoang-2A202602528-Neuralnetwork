"""Plot individual experiment histories and side-by-side metric comparisons.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    history = result["history"]
    cfg = result["cfg"]
    summary = result["summary"]
    epochs = history["epoch"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["val_acc"], label="accuracy")
    axes[1].plot(epochs, history["val_macro_f1"], label="macro-F1")
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score")
    axes[1].legend()
    axes[2].plot(epochs, history["grad_norm"], label="mean pre-clip norm")
    axes[2].set(title="Gradient norm", xlabel="Epoch", ylabel="L2 norm")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
        if summary.get("best_epoch") in epochs:
            axis.axvline(summary["best_epoch"], color="black", linestyle=":", alpha=0.6)
    fig.suptitle(
        f"{cfg.get('exp_id', 'experiment')} | {cfg.get('optimizer')} lr={cfg.get('lr')} "
        f"batch={cfg.get('batch')} hidden={cfg.get('hidden')} dropout={cfg.get('dropout')}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    if not results:
        raise ValueError("At least one result is required for a comparison plot.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 5))
    for result in results:
        history = result["history"]
        if metric not in history:
            raise KeyError(f"Metric {metric!r} is not present in experiment history.")
        axis.plot(history["epoch"], history[metric], marker=".", label=result["cfg"]["exp_id"])
    axis.set(title=title or metric, xlabel="Epoch", ylabel=metric)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
