import os
import requests
from tqdm import tqdm

MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
DEST_PATH = "app/models/RealESRGAN_x4plus.pth"

os.makedirs(os.path.dirname(DEST_PATH), exist_ok=True)

print("Downloading RealESRGAN_x4plus model...")
response = requests.get(MODEL_URL, stream=True)
total_size = int(response.headers.get('content-length', 0))

with open(DEST_PATH, 'wb') as f, tqdm(
    desc=DEST_PATH,
    total=total_size,
    unit='B',
    unit_scale=True,
    unit_divisor=1024,
) as bar:
    for data in response.iter_content(1024):
        f.write(data)
        bar.update(len(data))

print("✅ Download complete.")
