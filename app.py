import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
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
    100% Free • Studio Skin Glow & Lighting • Smart Head Positioning • Clean Signature
    </p>
""", unsafe_allow_html=True)

@st.cache_resource
def load_session():
    return new_session("silueta")

session = load_session()

def studio_face_retouch(pil_img):
    """Studio soft-box lighting, shadow lifting and portrait enhancement"""
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # 1. CLAHE Lighting Correction (Chehre ki chhanv hatane aur balanced light ke liye)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    retouched = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # 2. Skin Smoothness (Studio skin texture preserving blur)
    smooth = cv2.bilateralFilter(retouched, d=7, sigmaColor=35, sigmaSpace=35)
    
    # Blend natural texture with smoothness
    blended = cv2.addWeighted(retouched, 0.45, smooth, 0.55, 0)
    
    # 3. Sharpness & Eye Detail Boost
    gaussian = cv2.GaussianBlur(blended, (0, 0), 2.0)
    unsharp = cv2.addWeighted(blended, 1.35, gaussian, -0.35, 0)
    
    result_pil = Image.fromarray(cv2.cvtColor(unsharp, cv2.COLOR_BGR2RGB))
    
    # Final Brightness & Color Vibrancy Polish
    bright = ImageEnhance.Brightness(result_pil).enhance(1.10)
    contrast = ImageEnhance.Contrast(bright).enhance(1.08)
    color = ImageEnhance.Color(contrast).enhance(1.05)
    return color

def extract_clean_signature(sig_bytes, target_width=180):
    """Kagaz ka background remove karke sirf dark ink signature nikalna"""
    sig_img = Image.open(io.BytesIO(sig_bytes)).convert("L")
    sig_img = ImageOps.autocontrast(sig_img, cutoff=3)
    np_img = np.array(sig_img)
    
    # Thresholding for ink separation
    threshold = 180
    alpha = np.where(np_img < threshold, 255, 0).astype(np.uint8)
    
    h, w = np_img.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 15
    rgba[:, :, 1] = 15
    rgba[:, :, 2] = 15
    rgba[:, :, 3] = alpha
    
    clean_sig = Image.fromarray(rgba, mode="RGBA")
    bbox = clean_sig.getbbox()
    if bbox:
        clean_sig = clean_sig.crop(bbox)
        
    aspect = clean_sig.height / clean_sig.width
    sig_w = target_width
    sig_h = int(target_width * aspect)
    if sig_h > 80:
        sig_h = 80
        sig_w = int(sig_h / aspect)
        
    return clean_sig.resize((sig_w, sig_h), Image.Resampling.LANCZOS)

st.write("### 1️⃣ Target Photo Upload")
uploaded_photo = st.file_uploader("Passport ke liye photo upload karein", type=["jpg", "jpeg", "png", "webp"])

st.write("### 2️⃣ Signature Upload (Optional)")
uploaded_sign = st.file_uploader("Kagaz par kiye sign ki photo dalein (Agar lagana ho toh)", type=["jpg", "jpeg", "png", "webp"])

if uploaded_photo:
    if st.button("⚡ GENERATE STUDIO PASSPORT PHOTO NOW", type="primary", use_container_width=True):
        with st.spinner("Studio lighting, skin retouch aur solid background process ho raha hai..."):
            try:
                raw_img = Image.open(uploaded_photo).convert("RGB")
                
                # 1. Advanced Studio Face Retouch
                enhanced_person = studio_face_retouch(raw_img)
                
                # 2. Transparent Background Cutout
                cutout_pil = remove(enhanced_person, session=session).convert("RGBA")
                
                # 3. Studio Royal Sky-Blue Background (RGB: 140, 185, 235)
                bg = Image.new("RGBA", cutout_pil.size, (140, 185, 235, 255))
                composed = Image.alpha_composite(bg, cutout_pil).convert("RGB")
                
                # 4. Standard 3.5cm x 4.5cm Passport Cropping (413 x 531)
                target_w, target_h = 413, 531
                w, h = composed.size
                target_ratio = target_w / target_h
                current_ratio = w / h
                
                if current_ratio > target_ratio:
                    new_w = int(h * target_ratio)
                    left = (w - new_w) // 2
                    composed = composed.crop((left, 0, left + new_w, h))
                else:
                    new_h = int(w / target_ratio)
                    # Sir ke upar ka extra space kam karke face centered kiya (3% margin)
                    top = int((h - new_h) * 0.03)
                    composed = composed.crop((0, top, w, top + new_h))
                    
                composed = composed.resize((target_w, target_h), Image.Resampling.LANCZOS)
                
                # 5. Signature Overlay (Sirf tab lagega jab aap sign upload karenge)
                if uploaded_sign:
                    try:
                        clean_sig = extract_clean_signature(uploaded_sign.read(), target_width=180)
                        sig_x = int((target_w - clean_sig.width) / 2)
                        sig_y = int(target_h * 0.75)
                        composed.paste(clean_sig, (sig_x, sig_y), clean_sig)
                    except Exception as err:
                        st.warning(f"Signature load nahi hua: {err}")
                
                # 6. Studio White Border
                draw = ImageDraw.Draw(composed)
                draw.rounded_rectangle([(7, 7), (target_w - 7, target_h - 7)], radius=10, outline="white", width=4)
                
                # Output
                buf = io.BytesIO()
                composed.save(buf, format="JPEG", quality=98)
                out_bytes = buf.getvalue()
                
                st.success("✅ Studio Passport Photo Taiyaar!")
                st.image(out_bytes, caption="Studio Passport (3.5cm x 4.5cm)", width=290)
                
                st.download_button(
                    label="📥 DOWNLOAD PASSPORT PHOTO",
                    data=out_bytes,
                    file_name="studio_passport.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error: {repr(e)}")
