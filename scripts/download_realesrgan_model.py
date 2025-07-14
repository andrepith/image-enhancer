import os
import requests

def download_model():
    model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    model_path = "app/models/RealESRGAN_x4plus.pth"
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path):
        print("Model already downloaded.")
        return

    print("Downloading Real-ESRGAN model...")
    r = requests.get(model_url, stream=True)
    with open(model_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete.")

if __name__ == "__main__":
    download_model()
