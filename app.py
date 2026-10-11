import streamlit as st
import numpy as np
import cv2
import io
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="Twinkle Online Centre - Pro Passport Maker", layout="centered")

# --- SECRET FIX: Yahan 'u2netp' (Lite AI model) lagaya hai jisse free server HANG na ho ---
@st.cache_resource
def get_rembg_session():
    from rembg import new_session
    # u2netp specially low RAM aur fast speed ke liye hota hai
    return new_session("u2netp")

def auto_center_crop(image_cv):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    
    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]
        cx, cy = x + w // 2, y + h // 2
        cw, ch = int(w * 2.5), int(w * 2.5 * 1.33)
        x1, y1 = max(0, cx - cw // 2), max(0, cy - int(h * 0.8))
        return image_cv[y1:min(image_cv.shape[0], y1 + ch), x1:min(image_cv.shape[1], x1 + cw)]
    return image_cv

def enhance_image(image_cv):
    # Remini Jaisa Smooth Texture aur Glow
    smooth = cv2.bilateralFilter(image_cv, d=7, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced_cv = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)
    
    hsv = cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 10)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)

def process_passport(img_file, sig_file, sig_text):
    from rembg import remove
    
    # 1. Image Load & Fast Resize (Taaki server hang na ho)
    original = Image.open(img_file).convert("RGB")
    original.thumbnail((600, 600), Image.Resampling.LANCZOS)
    img_cv = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
    
    # 2. Crop
    cropped_cv = auto_center_crop(img_cv)
    
    # 3. Enhance
    enhanced_rgb = enhance_image(cropped_cv)
    enhanced_pil = Image.fromarray(enhanced_rgb)
    
    # 4. BG Remove (Lite fast session ke sath)
    session = get_rembg_session()
    img_byte_arr = io.BytesIO()
    enhanced_pil.save(img_byte_arr, format='PNG')
    bg_removed = remove(img_byte_arr.getvalue(), session=session)
    fg_img = Image.open(io.BytesIO(bg_removed)).convert("RGBA")
    
    # 5. Exact Blue Background Set Krna
    bg = Image.new("RGBA", fg_img.size, (100, 175, 230, 255))
    bg.paste(fg_img, (0, 0), fg_img)
    bg = bg.resize((600, 800), Image.LANCZOS)
    
    # 6. White Border
    final_img = bg.convert("RGB")
    draw = ImageDraw.Draw(final_img)
    border_width = 15
    draw.rectangle([border_width, border_width, final_img.width - border_width, final_img.height - border_width], outline="white", width=border_width)
    
    # 7. Signature Set karna
    if sig_file:
        sig_img = Image.open(sig_file).convert("RGB")
        sig_cv = cv2.cvtColor(np.array(sig_img), cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(sig_cv, 150, 255, cv2.THRESH_BINARY_INV)
        bold_sig = cv2.dilate(thresh, np.ones((2,2), np.uint8), iterations=1)
        rgba_sig = np.zeros((bold_sig.shape[0], bold_sig.shape[1], 4), dtype=np.uint8)
        rgba_sig[..., 3] = bold_sig
        rgba_sig[..., 0:3] = 0
        proc_sig = Image.fromarray(rgba_sig)
        
        sig_w = int(final_img.width * 0.7)
        sig_h = int(proc_sig.height * (sig_w / proc_sig.width))
        proc_sig = proc_sig.resize((sig_w, sig_h), Image.LANCZOS)
        final_img.paste(proc_sig, ((final_img.width - sig_w) // 2, final_img.height - sig_h - 30), proc_sig)
        
    elif sig_text:
        try: 
            font = ImageFont.truetype("BrushScriptMT.ttf", 60)
        except: 
            font = ImageFont.load_default()
        draw.text((final_img.width // 4, final_img.height - 100), sig_text, fill="black", font=font)
        
    return final_img

# --- UI Setup ---
st.title("📸 Twinkle Online Centre - Pro Passport Maker")
st.write("Super Fast AI Engine Loaded. App ab hang nahi hoga.")

col1, col2 = st.columns(2)
with col1: 
    photo_up = st.file_uploader("1. Photo Upload", type=['jpg', 'jpeg', 'png'])
with col2:
    sig_opt = st.radio("2. Sign Option:", ["Upload Image", "Type Name"])
    sig_up = st.file_uploader("Upload Sign", type=['jpg', 'jpeg', 'png']) if sig_opt == "Upload Image" else None
    sig_txt = st.text_input("Name:", "Twinkle Yadav") if sig_opt == "Type Name" else ""

if photo_up and st.button("Generate HD Passport"):
    with st.spinner("Processing in seconds (Memory Optimized)..."):
        result = process_passport(photo_up, sig_up, sig_txt)
        st.success("Photo Ready!")
        st.image(result, use_column_width=True)
        
        buf = io.BytesIO()
        result.save(buf, format="PNG", quality=100)
        st.download_button("Download HD Photo", data=buf.getvalue(), file_name="passport_hd.png", mime="image/png")
