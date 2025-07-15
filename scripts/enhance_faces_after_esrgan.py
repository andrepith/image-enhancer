import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

import cv2
import numpy as np
from PIL import Image
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from app.core.image_utils import show_side_by_side
from basicsr.archs.rrdbnet_arch import RRDBNet


def main(input_path, output_path):
    from app.core.image_utils import show_side_by_side

    print("Reading input image...")
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imread failed to read the input image")

    print("Initializing Real-ESRGAN...")
    esrgan = RealESRGANer(
        scale=2,
        model_path="app/models/RealESRGAN_x2plus.pth",
        model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
    )

    print("Running Real-ESRGAN...")
    img_up = esrgan.enhance(img, outscale=2)[0]
    print("Enhanced image shape:", img_up.shape)

    if img_up is None:
        raise ValueError("ESRGAN failed to return enhanced image.")

    print("Initializing GFPGAN...")
    restorer = GFPGANer(
        model_path="app/models/GFPGANv1.4.pth",
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )

    print("Running GFPGAN on enhanced image...")
    _, restored_list, _ = restorer.enhance(img_up, has_aligned=False, only_center_face=False, paste_back=True)

    if restored_list is None or len(restored_list) == 0:
        raise ValueError("GFPGAN did not return any enhanced faces.")

    restored_img = restored_list[0]
    restored_pil = Image.fromarray(restored_img)
    restored_pil.save(output_path)
    print(f"Saved final enhanced image to {output_path}")


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    main("data/raw/old_photo.jpg", "data/processed/final_enhanced.jpg")
