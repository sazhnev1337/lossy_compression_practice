import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def add_gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Returns np.ndarray: Noisy image, dtype uint8, clipped to [0, 255].
    """
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(loc=0.0, scale=sigma, size=image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(image: np.ndarray, density: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Returns np.ndarray: Noisy image, dtype uint8, clipped to [0, 255].
    """
    if rng is None:
        rng = np.random.default_rng()

    noisy = image.copy()
    n_pixels = image.size
    n_corrupt = int(n_pixels * density)

    # choose corrupted pixel indices without replacement
    indices = rng.choice(n_pixels, size=n_corrupt, replace=False)

    # first half → salt (255), second half → pepper (0)
    salt_indices   = indices[: n_corrupt // 2]
    pepper_indices = indices[n_corrupt // 2 :]

    noisy.flat[salt_indices]   = 255
    noisy.flat[pepper_indices] = 0

    return noisy


if __name__ == "__main__":

    img = np.array(Image.open("./media/cat512.jpg").convert("L"))


    rng = np.random.default_rng(seed=42)
    low    = add_gaussian_noise(img, sigma=10,  rng=rng)
    medium = add_gaussian_noise(img, sigma=25,  rng=rng)
    high   = add_gaussian_noise(img, sigma=50,  rng=rng)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, title, data in zip(
        axes,
        ["Original", "Low (σ=10)", "Medium (σ=25)", "High (σ=50)"],
        [img, low, medium, high],
    ):
        ax.imshow(data, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("./media/gaussian_noise_cover.png", dpi=150)
    plt.show()
