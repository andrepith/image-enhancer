# AI Image Enhancement Project

A powerful image enhancement tool that uses state-of-the-art AI models to upscale and restore old, low-quality photos. This project combines Real-ESRGAN for super-resolution and GFPGAN for face restoration to produce high-quality enhanced images.

## Features

- **Super-Resolution**: Upscale images by 2x or 4x using Real-ESRGAN
- **Face Restoration**: Enhance and restore faces in photos using GFPGAN
- **Combined Enhancement**: Apply both super-resolution and face restoration in sequence
- **Easy-to-use Scripts**: Simple command-line scripts for different enhancement tasks
- **Progress Tracking**: Visual progress bars for long-running enhancement operations
- **Side-by-side Comparison**: Built-in visualization to compare original and enhanced images

## Project Structure

```
├── app/
│   ├── core/
│   │   └── image_utils.py          # Image loading and visualization utilities
│   ├── models/
│   │   └── image_enhancer_model.py # Custom model implementations
│   └── api/
│       └── endpoints.py            # API endpoints (future feature)
├── scripts/
│   ├── download_*.py               # Model download scripts
│   ├── enhance_image.py            # Basic image enhancement
│   ├── enhance_with_faces.py       # Face-focused enhancement
│   ├── enhance_faces_after_esrgan.py # Combined enhancement pipeline
│   └── preview_images.py           # Image comparison tool
├── data/
│   ├── raw/                        # Input images
│   ├── processed/                  # Enhanced output images
│   └── high_quality/               # Reference high-quality images
└── tests/                          # Unit tests
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd image-enhancement-project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchvision
   pip install realesrgan gfpgan
   pip install pillow matplotlib opencv-python tqdm requests
   pip install basicsr
   ```

4. **Download AI models**
   ```bash
   # Download Real-ESRGAN x4 model
   python scripts/download_realesrgan_model.py
   
   # Download Real-ESRGAN x2 model
   python scripts/download_esrgan_x2_model.py
   
   # Download GFPGAN face restoration model
   python scripts/download_gfpgan_model.py
   ```

## Usage

### Basic Image Enhancement

Enhance any image with 4x super-resolution:

```bash
python scripts/enhance_image.py
```

This will process `data/raw/old_photo.jpg` and save the result to `data/processed/enhanced_photo.jpg`.

### Face-Focused Enhancement

For photos with faces, use the face restoration model:

```bash
python scripts/enhance_with_faces.py
```

### Combined Enhancement Pipeline

For the best results on photos with faces, use the combined approach:

```bash
python scripts/enhance_faces_after_esrgan.py
```

This script:
1. First applies Real-ESRGAN 2x super-resolution
2. Then applies GFPGAN face restoration on the upscaled image
3. Saves the final enhanced result

### Preview and Compare Images

Compare original and reference images:

```bash
python scripts/preview_images.py
```

## Configuration

### Input/Output Paths

Most scripts are pre-configured with these paths:
- **Input**: `data/raw/old_photo.jpg`
- **Output**: `data/processed/[enhanced_filename].jpg`

You can modify the paths directly in each script or extend them to accept command-line arguments.

### Model Parameters

The enhancement scripts use these default settings:

**Real-ESRGAN**:
- Scale: 4x (or 2x for combined pipeline)
- Tile: 0 (no tiling for memory management)
- Half precision: False (for better quality)

**GFPGAN**:
- Upscale: 2x
- Architecture: "clean"
- Channel multiplier: 2

## Models

This project uses pre-trained models:

- **RealESRGAN_x4plus.pth**: 4x super-resolution model (~67MB)
- **RealESRGAN_x2plus.pth**: 2x super-resolution model (~67MB)  
- **GFPGANv1.4.pth**: Face restoration model (~348MB)

Models are automatically downloaded to `app/models/` when you run the download scripts.

## Examples

### Before and After

The enhanced images will show significant improvements in:
- **Resolution**: 4x increase in pixel count
- **Sharpness**: Enhanced edge definition and detail
- **Face Quality**: Restored facial features and skin texture
- **Overall Clarity**: Reduced noise and artifacts

## Development

### Adding New Enhancement Methods

1. Create a new script in `scripts/`
2. Use the utilities in `app/core/image_utils.py` for image loading and visualization
3. Follow the existing pattern for progress tracking and error handling

### Testing

Run tests with:
```bash
python -m pytest tests/
```

### API Development

The project structure includes an `app/api/` directory for future REST API development.

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce tile size or use half precision
2. **Model Download Fails**: Check internet connection and retry
3. **CUDA Errors**: Set `half=False` in model initialization
4. **Import Errors**: Ensure all dependencies are installed in the virtual environment

### Performance Tips

- Use GPU acceleration when available (CUDA)
- Enable half precision for faster processing: `half=True`
- Use tiling for very large images: `tile=400`

## License

This project uses pre-trained models from:
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [GFPGAN](https://github.com/TencentARC/GFPGAN)

Please refer to their respective licenses for usage terms.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- **Real-ESRGAN** team for the super-resolution models
- **GFPGAN** team for the face restoration technology
- **BasicSR** for the underlying framework
