import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps
import io
import cv2
import numpy as np
from rembg import remove, new_session

st.set_page_config(page_title="Studio Passport AI Maker", page_icon="📸", layout="centered")

ACCESS_CODE = "1234"

if "unlocked" not in st.session_state:
    st.session_state.unlocked = False

if not st.session_state.unlocked:
    st.subheader("🔒 Studio Safe-Gate")
    entered_code = st.text_input("Enter Access Code:", type="password")
    if st.button("Unlock Studio Engine", type="primary", use_container_width=True):
        if entered_code == ACCESS_CODE:
            st.session_state.unlocked = True
            st.rerun()
        else:
            st.error("Galat Code! Sahi passcode enter karein.")
    st.stop()

st.markdown("""
    <h2 style='text-align: center; color: #0d6efd;'>📸 Studio Passport Photo AI Maker</h2>
    <p style='text-align: center; color: gray; font-size: 13px;'>
    Auto Face Centering • Studio Lighting • Cursive Signature • Zero Billing
    </p>
""", unsafe_allow_html=True)

@st.cache_resource
def load_session():
    return new_session("silueta")

session = load_session()

def create_studio_gradient_bg(w, h):
    """Studio soft spotlight sky-blue background"""
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    center_x, center_y = w // 2, int(h * 0.40)
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2) / (w * 0.75)
            factor = min(1.0, dist)
            # Center bright sky-blue to edges royal-blue
            r = int(160 * (1 - factor) + 115 * factor)
            g = int(195 * (1 - factor) + 160 * factor)
            b = int(240 * (1 - factor) + 215 * factor)
            bg[y, x] = [b, g, r]
    return Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)).convert("RGBA")

def smart_face_crop(pil_img, cutout_pil):
    """Face detect karke exact photo studio jaisa crop aur center karna"""
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    
    img_w, img_h = pil_img.size
    target_w, target_h = 413, 531
    aspect = target_w / target_h
    
    if len(faces) > 0:
        # Sabse bada chehra select karein
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_cx = fx + fw // 2
        
        # Passport standards: Head height photo ki lagbhag 50-55% honi chahiye
        crop_h = int(fh / 0.52)
        crop_w = int(crop_h * aspect)
        
        # Sir ke upar standard 15% studio gap
        top = int(fy - (crop_h * 0.16))
        left = int(face_cx - (crop_w // 2))
        
        # Bounds checking
        if crop_w > img_w:
            crop_w = img_w
            crop_h = int(crop_w / aspect)
            left = 0
            top = max(0, fy - int(crop_h * 0.15))
        else:
            left = max(0, min(left, img_w - crop_w))
            top = max(0, min(top, img_h - crop_h))
            
        cropped = cutout_pil.crop((left, top, left + crop_w, top + crop_h))
    else:
        # Fallback agar face sideways ho
        bbox = cutout_pil.getbbox()
        if bbox:
            bx0, by0, bx1, by1 = bbox
            bw, bh = bx1 - bx0, by1 - by0
            crop_h = int(bh * 1.1)
            crop_w = int(crop_h * aspect)
            left = max(0, (bx0 + bx1)//2 - crop_w // 2)
            top = max(0, by0 - int(bh * 0.08))
            cropped = cutout_pil.crop((left, top, min(img_w, left + crop_w), min(img_h, top + crop_h)))
        else:
            cropped = cutout_pil
            
    return cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)

st.write("### 1️⃣ Target Photo Upload")
uploaded_photo = st.file_uploader("Passport ke liye photo dalein", type=["jpg", "jpeg", "png", "webp"])

st.write("### 2️⃣ Signature Settings")
col1, col2 = st.columns(2)
with col1:
    sig_mode = st.radio("Signature Type:", ["✏️ Typed Cursive Name", "📷 Upload Signature Photo", "❌ No Signature"], index=0)
with col2:
    if sig_mode == "✏️ Typed Cursive Name":
        typed_sig = st.text_input("Signature Name:", value="Pankaj Yadav")
    elif sig_mode == "📷 Upload Signature Photo":
        sig_file = st.file_uploader("Kagaz par sign ki hui photo", type=["jpg", "png", "jpeg"])

if uploaded_photo:
    if st.button("⚡ GENERATE STUDIO PASSPORT PHOTO NOW", type="primary", use_container_width=True):
        with st.spinner("Face detect karke Studio Passport Photo banayi ja rahi hai..."):
            try:
                raw_img = Image.open(uploaded_photo).convert("RGB")
                
                # 1. Subtle lighting boost
                enh_bright = ImageEnhance.Brightness(raw_img).enhance(1.05)
                enh_contrast = ImageEnhance.Contrast(enh_bright).enhance(1.05)
                enh_sharp = ImageEnhance.Sharpness(enh_contrast).enhance(1.20)
                
                # 2. Background removal
                cutout_pil = remove(enh_sharp, session=session).convert("RGBA")
                
                # 3. Exact Face Centering & Passport Cropping
                person_cropped = smart_face_crop(raw_img, cutout_pil)
                
                # 4. Studio Vignette Gradient Background (413 x 531)
                bg = create_studio_gradient_bg(413, 531)
                composed = Image.alpha_composite(bg, person_cropped).convert("RGB")
                
                # 5. Signature Overlay
                draw = ImageDraw.Draw(composed)
                if sig_mode == "✏️ Typed Cursive Name" and typed_sig.strip():
                    try:
                        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
                    except Exception:
                        font = ImageFont.load_default()
                    draw.text((115, 410), typed_sig.strip(), fill=(15, 15, 15), font=font)
                    
                elif sig_mode == "📷 Upload Signature Photo" and sig_file:
                    sig_img = Image.open(sig_file).convert("L")
                    sig_img = ImageOps.autocontrast(sig_img, cutoff=2)
                    np_sig = np.array(sig_img)
                    alpha = np.where(np_sig < 175, 255, 0).astype(np.uint8)
                    h, w = np_sig.shape
                    rgba = np.zeros((h, w, 4), dtype=np.uint8)
                    rgba[:, :, :3] = 15
                    rgba[:, :, 3] = alpha
                    clean_sig = Image.fromarray(rgba, mode="RGBA")
                    bbox = clean_sig.getbbox()
                    if bbox:
                        clean_sig = clean_sig.crop(bbox)
                    clean_sig = clean_sig.resize((170, int(170 * clean_sig.height / clean_sig.width)), Image.Resampling.LANCZOS)
                    composed.paste(clean_sig, (120, 410), clean_sig)

                # 6. Rounded White Border
                draw.rounded_rectangle([(6, 6), (406, 524)], radius=12, outline="white", width=4)
                
                buf = io.BytesIO()
                composed.save(buf, format="JPEG", quality=98)
                out_bytes = buf.getvalue()
                
                st.success("✅ Studio Passport Photo Taiyaar!")
                st.image(out_bytes, caption="Studio Passport Photo (3.5cm x 4.5cm)", width=290)
                
                st.download_button(
                    label="📥 DOWNLOAD PASSPORT PHOTO",
                    data=out_bytes,
                    file_name="studio_passport.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error: {repr(e)}")
