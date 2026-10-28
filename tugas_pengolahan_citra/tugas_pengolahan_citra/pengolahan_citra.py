import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from pathlib import Path

# Pastikan gambar.jpg berada di folder yang sama dengan program ini.
INPUT_IMAGE = "gambar.jpg"
OUTPUT_IMAGE = "hasil_peningkatan.png"

def to_grayscale(image):
    """Mengubah gambar RGB/RGBA menjadi grayscale jika diperlukan."""
    if image.ndim == 3:
        image = image[..., :3]  # Abaikan kanal alpha jika ada
        image = (0.299 * image[..., 0] +
                 0.587 * image[..., 1] +
                 0.114 * image[..., 2])
    return image.astype(np.float32)

def main():
    if not Path(INPUT_IMAGE).exists():
        raise FileNotFoundError(
            f"File {INPUT_IMAGE} tidak ditemukan. "
            "Letakkan gambar di folder yang sama dengan program."
        )

    # 1. Membaca citra menggunakan imageio
    original = iio.imread(INPUT_IMAGE)
    gray = to_grayscale(original)

    # 2. Normalisasi nilai piksel ke rentang 0-255 menggunakan NumPy
    min_val, max_val = gray.min(), gray.max()
    if max_val > min_val:
        normalized = (gray - min_val) / (max_val - min_val) * 255.0
    else:
        normalized = gray.copy()

    # 3. Mengurangi noise dengan median filter dari scipy.ndimage
    denoised = ndimage.median_filter(normalized, size=3)

    # 4. Mempertajam detail dengan unsharp masking
    blurred = ndimage.gaussian_filter(denoised, sigma=1.0)
    sharpened = np.clip(denoised + 1.2 * (denoised - blurred), 0, 255)

    # 5. Simpan hasil citra yang ditingkatkan
    enhanced = sharpened.astype(np.uint8)
    iio.imwrite(OUTPUT_IMAGE, enhanced)

    # 6. Tampilkan perbandingan citra awal dan hasil peningkatan
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Citra Awal (Grayscale)")
    axes[0].axis("off")

    axes[1].imshow(enhanced, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Citra Setelah Peningkatan")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("screenshot_hasil.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Program selesai.")
    print(f"Ukuran citra: {gray.shape[1]} x {gray.shape[0]} piksel")
    print(f"Hasil tersimpan sebagai: {OUTPUT_IMAGE}")
    print("Metode: normalisasi kontras, median filter, dan unsharp masking.")

if __name__ == "__main__":
    main()
