from PIL import Image
import matplotlib.pyplot as plt
import os

def load_image(path, size=(256, 256)):
    """Load an image and resize it."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    return img

def show_side_by_side(img1, img2, title1="Before", title2="After"):
    """Display two images side by side."""
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img1)
    axs[0].set_title(title1)
    axs[0].axis("off")
    
    axs[1].imshow(img2)
    axs[1].set_title(title2)
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()
