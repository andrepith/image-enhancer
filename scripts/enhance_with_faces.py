import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from PIL import Image
from gfpgan import GFPGANer
from app.core.image_utils import show_side_by_side

def enhance_faces(input_path, output_path):
    model_path = "app/models/GFPGANv1.4.pth"

    # Init GFPGAN
    restorer = GFPGANer(
        model_path=model_path,
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None
    )

    # Load image
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Enhance
    result = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    restored_img = result[1]


    # Save output
    restored_pil = Image.fromarray(restored_img[0])
    restored_pil.save(output_path)

    # Compare
    original = Image.fromarray(img)
    show_side_by_side(original, restored_pil, title1="Original", title2="Face Enhanced")

if __name__ == "__main__":
    enhance_faces("data/raw/old_photo_2.jpg", "data/processed/enhanced_with_faces_2.jpg")
