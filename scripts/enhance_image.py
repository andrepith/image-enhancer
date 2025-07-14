import torch
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from app.core.image_utils import show_side_by_side

def enhance_image(input_path, output_path):
    # Initialize model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)
    
    upsampler = RealESRGANer(
        scale=4,
        model_path='app/models/RealESRGAN_x4plus.pth',
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False  # Use True if using CUDA with float16 support
    )

    img = Image.open(input_path).convert('RGB')
    output, _ = upsampler.enhance(np.array(img), outscale=1)
    
    enhanced_img = Image.fromarray(output)
    enhanced_img.save(output_path)
    
    # Visualize comparison
    show_side_by_side(img, enhanced_img, title1="Old", title2="Enhanced")

if __name__ == "__main__":
    import numpy as np
    enhance_image("data/raw/old_photo.jpg", "data/processed/enhanced_photo.jpg")
