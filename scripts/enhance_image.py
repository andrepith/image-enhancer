import torch
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from app.core.image_utils import show_side_by_side
from tqdm import tqdm
import time

def enhance_image(input_path, output_path):
    # Create progress bar for the entire process
    with tqdm(total=100, desc="Image Enhancement", unit="%") as pbar:
        
        # Initialize model
        pbar.set_description("Loading model...")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        pbar.update(20)
        
        # Initialize upsampler
        pbar.set_description("Initializing upsampler...")
        upsampler = RealESRGANer(
            scale=4,
            model_path='app/models/RealESRGAN_x4plus.pth',
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False  # Use True if using CUDA with float16 support
        )
        pbar.update(20)
        
        # Load image
        pbar.set_description("Loading image...")
        img = Image.open(input_path).convert('RGB')
        pbar.update(10)
        
        # Enhance image (this is the main processing step)
        pbar.set_description("Enhancing image...")
        output, _ = upsampler.enhance(np.array(img), outscale=1)
        pbar.update(30)
        
        # Save enhanced image
        pbar.set_description("Saving enhanced image...")
        enhanced_img = Image.fromarray(output)
        enhanced_img.save(output_path)
        pbar.update(10)
        
        # Visualize comparison
        pbar.set_description("Generating comparison...")
        show_side_by_side(img, enhanced_img, title1="Old", title2="Enhanced")
        pbar.update(10)
        
        pbar.set_description("Complete!")

def enhance_image_with_detailed_progress(input_path, output_path):
    """Alternative version with more detailed progress tracking"""
    
    print("🚀 Starting image enhancement process...")
    
    # Step 1: Model initialization
    with tqdm(desc="Loading model", unit="step") as pbar:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        pbar.update(1)
    
    # Step 2: Upsampler initialization
    with tqdm(desc="Initializing upsampler", unit="step") as pbar:
        upsampler = RealESRGANer(
            scale=4,
            model_path='app/models/RealESRGAN_x4plus.pth',
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False
        )
        pbar.update(1)
    
    # Step 3: Load and process image
    with tqdm(desc="Loading image", unit="step") as pbar:
        img = Image.open(input_path).convert('RGB')
        pbar.update(1)
    
    # Step 4: Enhancement (with simulated progress for demonstration)
    with tqdm(desc="Enhancing image", unit="iteration", total=100) as pbar:
        # Since upsampler.enhance doesn't provide progress callback,
        # we simulate progress updates
        def enhance_with_progress():
            # Start enhancement in a separate thread/process if needed
            # For now, we'll simulate the progress
            for i in range(100):
                time.sleep(0.01)  # Simulate processing time
                pbar.update(1)
                if i == 99:  # At the end, do the actual enhancement
                    output, _ = upsampler.enhance(np.array(img), outscale=1)
                    return output
        
        # For actual implementation, replace the above with:
        output, _ = upsampler.enhance(np.array(img), outscale=1)
        pbar.update(100)
    
    # Step 5: Save results
    with tqdm(desc="Saving enhanced image", unit="step") as pbar:
        enhanced_img = Image.fromarray(output)
        enhanced_img.save(output_path)
        pbar.update(1)
    
    # Step 6: Show comparison
    with tqdm(desc="Generating comparison", unit="step") as pbar:
        show_side_by_side(img, enhanced_img, title1="Old", title2="Enhanced")
        pbar.update(1)
    
    print("✅ Image enhancement completed successfully!")

if __name__ == "__main__":
    enhance_image("data/raw/old_photo.jpg", "data/processed/enhanced_photo.jpg")
    
    # Alternative: use the detailed version
    # enhance_image_with_detailed_progress("data/raw/old_photo.jpg", "data/processed/enhanced_photo.jpg")