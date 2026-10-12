import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from rembg import remove
import io

st.set_page_config(page_title="AI Studio Passport Maker", layout="wide")

def create_studio_bg(w, h):
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h * 0.35
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    norm = np.clip(dist / np.sqrt(cx**2 + cy**2), 0.0, 1.0)
    
    c_center = np.array([215, 238, 255])
    c_edge = np.array([12, 60, 125])
    
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(3):
        bg[:, :, i] = (c_center[i] * (1.0 - norm) + c_edge[i] * norm).astype(np.uint8)
    return Image.fromarray(bg).convert("RGBA")

def align_and_crop_face(pil_img):
    img_np = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    if len(faces) == 0:
        return pil_img
        
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    cx, cy = fx + fw // 2, fy + fh // 2
    
    target_h = int(fh * 1.85)
    target_w = int(target_h * (3.5 / 4.5))
    
    x1 = max(0, cx - target_w // 2)
    y1 = max(0, int(cy - target_h * 0.42))
    x2 = min(pil_img.width, x1 + target_w)
    y2 = min(pil_img.height, y1 + target_h)
    
    return pil_img.crop((x1, y1, x2, y2))

def clean_signature(sign_img):
    gray = np.array(sign_img.convert("L"))
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 7)
    kernel = np.ones((2, 2), np.uint8)
    bold = cv2.dilate(binary, kernel, iterations=1)
    
    rgba = np.zeros((bold.shape[0], bold.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 0] = 5
    rgba[:, :, 1] = 5
    rgba[:, :, 2] = 5
    rgba[:, :, 3] = bold
    return Image.fromarray(rgba, "RGBA")

def generate_print_sheet(photo_img):
    sheet_w, sheet_h = 1800, 1200 # 6x4 inch at 300 DPI
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    
    p_w, p_h = 413, 531 # standard passport size
    p_resized = photo_img.resize((p_w, p_h), Image.Resampling.LANCZOS)
    
    # 4x2 grid (8 photos)
    spacing_x = (sheet_w - (4 * p_w)) // 5
    spacing_y = (sheet_h - (2 * p_h)) // 3
    
    for row in range(2):
        for col in range(4):
            x = spacing_x + col * (p_w + spacing_x)
            y = spacing_y + row * (p_h + spacing_y)
            sheet.paste(p_resized, (x, y))
    return sheet

st.title("AI Passport Studio SaaS Tool")
col1, col2 = st.columns(2)

with col1:
    photo_file = st.file_uploader("Upload Customer Photo", type=["jpg", "png", "jpeg"])
    sign_file = st.file_uploader("Upload Signature", type=["jpg", "png", "jpeg"])
    process_btn = st.button("Generate Passport & Print Sheet")

with col2:
    if process_btn and photo_file:
        with st.spinner("AI Processing..."):
            raw_photo = Image.open(photo_file)
            aligned = align_and_crop_face(raw_photo)
            
            # Matting
            cutout = remove(aligned)
            
            # Studio BG
            w, h = 600, 800
            bg = create_studio_bg(w, h)
            cutout_resized = cutout.resize((w, h), Image.Resampling.LANCZOS)
            combined = Image.alpha_composite(bg, cutout_resized)
            
            # Enhance
            enhancer = ImageEnhance.Sharpness(combined.convert("RGB"))
            enhanced = enhancer.enhance(1.25)
            contrast = ImageEnhance.Contrast(enhanced)
            final_img = contrast.enhance(1.08).convert("RGBA")
            
            # Sign Overlay
            if sign_file:
                raw_sign = Image.open(sign_file)
                clean_sign = clean_signature(raw_sign)
                
                target_w = int(w * 0.45)
                ratio = target_w / clean_sign.width
                clean_sign = clean_sign.resize((target_w, int(clean_sign.height * ratio)), Image.Resampling.LANCZOS)
                
                pos_x = w - target_w - 25
                pos_y = h - clean_sign.height - 35
                final_img.paste(clean_sign, (pos_x, pos_y), clean_sign)
            
            single_result = final_img.convert("RGB")
            sheet_result = generate_print_sheet(single_result)
            
            st.image(single_result, caption="Single Passport Photo", width=300)
            
            buf_single = io.BytesIO()
            single_result.save(buf_single, format="JPEG", quality=95)
            st.download_button("Download Single Photo", buf_single.getvalue(), "passport.jpg", "image/jpeg")
            
            buf_sheet = io.BytesIO()
            sheet_result.save(buf_sheet, format="JPEG", quality=95)
            st.download_button("Download 4x6 Print Sheet (8 Photos)", buf_sheet.getvalue(), "print_sheet_4x6.jpg", "image/jpeg")
