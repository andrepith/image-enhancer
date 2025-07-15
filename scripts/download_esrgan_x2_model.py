import os
import requests

URL = "https://huggingface.co/dtarnow/UPscaler/resolve/main/RealESRGAN_x2plus.pth"
DEST = "app/models/RealESRGAN_x2plus.pth"

os.makedirs(os.path.dirname(DEST), exist_ok=True)
print("Downloading RealESRGAN x2plus model...")
r = requests.get(URL, stream=True)
with open(DEST, "wb") as f:
    for chunk in r.iter_content(8192):
        f.write(chunk)
print("Done.")
