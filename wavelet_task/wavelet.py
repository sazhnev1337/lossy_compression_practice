import numpy as np
import pywt
from PIL import Image
import matplotlib.pyplot as plt

from noise import add_gaussian_noise, add_salt_and_pepper_noise


def threshold_coeffs(coeffs: list, threshold: float) -> list:
    thresholded = [coeffs[0]]  # keep untouched
    for level_detail in coeffs[1:]:
        thresholded.append(
            tuple(pywt.threshold(subband, threshold, mode="hard") for subband in level_detail)
        )
    return thresholded


def reconstruct(coeffs: list, wavelet: str) -> np.ndarray:
    restored = pywt.waverec2(coeffs, wavelet=wavelet)
    return np.clip(restored, 0, 255).astype(np.uint8)


def rmse(original: np.ndarray, restored: np.ndarray) -> float:
    """Root Mean Square Error between original and restored image."""
    diff = original.astype(np.float64) - restored.astype(np.float64)
    return np.sqrt(np.mean(diff ** 2))


if __name__ == "__main__":
    WAVELET    = "haar"
    LEVEL      = 4
    NOISE_TYPE = "gaussian"   # "gaussian" | "salt_and_pepper"
    # NOISE_TYPE = "salt_and_pepper"   # "gaussian" | "salt_and_pepper"

    # gaussian:        sigma   low=10, medium=25, high=50
    # salt_and_pepper: density low=0.02, medium=0.05, high=0.15
    SIGMA   = 10
    DENSITY = 0.15

    img = np.array(Image.open("./media/cover512gray.jpg").convert("L"))

    rng = np.random.default_rng(seed=42)
    if NOISE_TYPE == "gaussian":
        noisy    = add_gaussian_noise(img, sigma=SIGMA, rng=rng)
        noise_tag = f"gauss_sig{SIGMA}"
    elif NOISE_TYPE == "salt_and_pepper":
        noisy    = add_salt_and_pepper_noise(img, density=DENSITY, rng=rng)
        noise_tag = f"snp_d{DENSITY}"


    thresholds = np.arange(0, 100, 2)
    errors = []
    for t in thresholds:

        coeffs   = pywt.wavedec2(noisy.astype(np.float64), WAVELET, LEVEL)
        coeffs_t = threshold_coeffs(coeffs, threshold=t)
        restored = reconstruct(coeffs_t, wavelet=WAVELET)
        errors.append(rmse(img, restored))

    best_t    = thresholds[np.argmin(errors)]
    best_rmse = min(errors)
    print(f"Best threshold: {best_t}, RMSE: {best_rmse:.2f}")

    # --- RMSE vs threshold ---
    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, errors)
    plt.axvline(best_t, color="red", linestyle="--", label=f"best t={best_t}")
    plt.xlabel("Threshold")
    plt.ylabel("RMSE")
    plt.title(f"RMSE vs threshold  |  {WAVELET}, level={LEVEL}, {noise_tag}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./media/rmse_{noise_tag}.png", dpi=150)
    plt.show()

    # --- Сохранение каждого порога отдельным файлом ---
    import os
    preview_thresholds = [0, 10, 20, best_t, 50, 80]

    out_dir = f"./media/thresholds_{noise_tag}"
    os.makedirs(out_dir, exist_ok=True)

    Image.fromarray(img).save(f"{out_dir}/00_original.png")
    Image.fromarray(noisy).save(f"{out_dir}/01_noisy_{noise_tag}.png")

    for t in preview_thresholds:
        c   = pywt.wavedec2(noisy.astype(np.float64), WAVELET, LEVEL)
        c   = threshold_coeffs(c, threshold=t)
        r   = reconstruct(c, wavelet=WAVELET)
        err = rmse(img, r)
        tag = "_BEST" if t == best_t else ""
        fname = f"{out_dir}/t{t:03d}_rmse{err:.1f}{tag}.png"
        Image.fromarray(r).save(fname)
