from app.core.image_utils import load_image, show_side_by_side

# Adjust these paths if needed
low_res_path = "data/raw/old_photo.jpg"
high_res_path = "data/high_quality/good_photo.jpg"

low_img = load_image(low_res_path)
high_img = load_image(high_res_path)

show_side_by_side(low_img, high_img, title1="Old / Low-Res", title2="Modern / High-Res")
