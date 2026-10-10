import streamlit as st
import numpy as np
import cv2
import io
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="Twinkle Online Centre - Pro Passport Maker", layout="centered")

# AI session ko memory mein save rakhega taaki baar-baar load na ho (Super Fast Loading)
@st.cache_resource
def get_rembg_session():
    from rembg import new_session
    return new_session("u2net")

def make_passport(img_file, sig_file, sig_text):
    from rembg import remove
    
    # 1. Image Load & Fast Resize (Speed ke liye)
    original = Image.open(img_file).convert("RGB")
    original.thumbnail((800, 800), Image.Resampling.LANCZOS)
    
    img_cv = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 2. Auto Center Crop
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    
    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]
        cx, cy = x + w // 2, y + h // 2
        cw, ch = int(w * 2.5), int(w * 2.5 * 1.33)
        x1, y1 = max(0, cx - cw // 2), max(0, cy - int(h * 0.8))
        img_cv = img_cv[y1:min(img_cv.shape[0], y1 + ch), x1:min(img_cv.shape[1], x1 + cw)]
    
    # 3. Remini-like Enhancement (Smooth Texture & Glow)
    smooth = cv2.bilateralFilter(img_cv, d=7, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced_cv = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)
    
    hsv = cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 10)
    final_cv = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)
    
    enhanced_pil = Image.fromarray(final_cv)
    
    # 4. BG Remove & Exactly Blue BG Set
    session = get_rembg_session()
    img_byte_arr = io.BytesIO()
    enhanced_pil.save(img_byte_arr, format='PNG')
    bg_removed = remove(img_byte_arr.getvalue(), session=session)
    fg_img = Image.open(io.BytesIO(bg_removed)).convert("RGBA")
    
    bg = Image.new("RGBA", fg_img.size, (100, 175, 230, 255)) # Aapka blue color
    bg.paste(fg_img, (0, 0), fg_img)
    bg = bg.resize((600, 800), Image.LANCZOS)
    
    # 5. White Border (Aapke reference image jaisa)
    final_img = bg.convert("RGB")
    draw = ImageDraw.Draw(final_img)
    border_width = 15
    draw.rectangle([border_width, border_width, final_img.width - border_width, final_img.height - border_width], outline="white", width=border_width)
    
    # 6. Bold & Transparent Signature
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

st.title("📸 Twinkle Online Centre - Pro Passport Maker")

col1, col2 = st.columns(2)
with col1: 
    photo_up = st.file_uploader("1. Photo Upload (Kaisi bhi size ki)", type=['jpg', 'jpeg', 'png'])
with col2:
    sig_opt = st.radio("2. Sign Option:", ["Upload Image", "Type Name"])
    sig_up = st.file_uploader("Upload Sign", type=['jpg', 'jpeg', 'png']) if sig_opt == "Upload Image" else None
    sig_txt = st.text_input("Name:", "Pankaj Yadav") if sig_opt == "Type Name" else ""

if photo_up and st.button("Generate HD Passport"):
    with st.spinner("AI Processing Start (Fast Mode)..."):
        result = make_passport(photo_up, sig_up, sig_txt)
        st.success("Ekdum Kadak Photo Ready!")
        st.image(result, use_column_width=True)
        
        buf = io.BytesIO()
        result.save(buf, format="PNG", quality=100)
        st.download_button("Download HD Photo", data=buf.getvalue(), file_name="passport.png", mime="image/png")
