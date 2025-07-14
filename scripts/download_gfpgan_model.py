import os
import requests

def download_model():
    model_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
    model_path = "app/models/GFPGANv1.4.pth"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path):
        print("Model already exists.")
        return

    print("Downloading GFPGAN model...")
    r = requests.get(model_url, stream=True)
    with open(model_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

if __name__ == "__main__":
    download_model()
