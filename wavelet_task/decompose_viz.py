import numpy as np
import pywt
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def build_decomposition_image(coeffs: list) -> np.ndarray:
    
    n_levels = len(coeffs) - 1
    cA = coeffs[0]
    h, w = cA.shape[0] * (2 ** n_levels), cA.shape[1] * (2 ** n_levels)
    canvas = np.zeros((h, w), dtype=np.float64)

    def normalize(arr: np.ndarray) -> np.ndarray:
        """Scale array to [0, 255] for display."""
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn) * 255

    canvas[: cA.shape[0], : cA.shape[1]] = np.clip(cA, 0, 255)

    size = cA.shape[0] 
    for i, (cH, cV, cD) in enumerate(coeffs[1:], start=1):
        row0 = 0
        col0 = size
        canvas[row0 : row0 + size, col0 : col0 + size * 2 // 2] = normalize(cH)  # top-right
        canvas[size : size * 2, : size] = normalize(cV)                          # bottom-left
        canvas[size : size * 2, size : size * 2] = normalize(cD)                 # bottom-right
        size *= 2  

    return canvas


def annotate_subbands(ax, coeffs: list) -> None:
    """Подписи квадратов"""
    n_levels = len(coeffs) - 1
    cA = coeffs[0]
    s = cA.shape[0]  # size of coarsest subband

    labels = []
    # Approximation
    labels.append((s / 2, s / 2, f"cA{n_levels}"))
    # Details per level
    size = s
    for lvl in range(n_levels, 0, -1):
        labels.append((size / 2,           size + size / 2,  f"cH{lvl}"))
        labels.append((size + size / 2,    size / 2,         f"cV{lvl}"))
        labels.append((size + size / 2,    size + size / 2,  f"cD{lvl}"))
        size *= 2

    for row, col, text in labels:
        ax.text(col, row, text, color="red", fontsize=8,
                ha="center", va="center", fontweight="bold")


if __name__ == "__main__":
    WAVELET = "haar"
    LEVEL   = 2

    img = np.array(Image.open("./media/cover512gray.jpg").convert("L"))

    coeffs = pywt.wavedec2(img.astype(np.float64), wavelet=WAVELET, level=LEVEL)

    canvas = build_decomposition_image(coeffs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(canvas, cmap="gray")
    axes[1].set_title(f"Wavelet decomposition ({WAVELET}, {LEVEL} levels)")
    axes[1].axis("off")
    annotate_subbands(axes[1], coeffs)

    plt.tight_layout()
    plt.savefig("./media/decomposition.png", dpi=150)
    plt.show()