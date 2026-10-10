import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
import io
import numpy as np
from rembg import remove, new_session

st.set_page_config(page_title="Studio Passport AI Maker", page_icon="📸", layout="centered")

# --- 🔒 PASSCODE ---
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
    100% Free • Perfect Passport Framing • Natural Studio Lighting • Auto Sign Transparent
    </p>
""", unsafe_allow_html=True)

@st.cache_resource
def load_session():
    return new_session("silueta")

session = load_session()

def natural_studio_lighting(pil_img):
    """Natural studio strobe-light balancing without dark or cartoonish artifacts"""
    # 1. Chehre ki lighting balance (Shadow lift bina saturation jalaye)
    bright = ImageEnhance.Brightness(pil_img).enhance(1.06)
    # 2. Gentle contrast taaki natural depth bani rahe
    contrast = ImageEnhance.Contrast(bright).enhance(1.04)
    # 3. Clean portrait sharpness (eyes, hair, lips crisp karne ke liye)
    sharp = ImageEnhance.Sharpness(contrast).enhance(1.20)
    # 4. Subtle color vibrancy
    color = ImageEnhance.Color(sharp).enhance(1.02)
    return color

def process_signature(sig_bytes, target_w=190):
    """Kagaz ka background remove karke sirf dark ink sign nikalna"""
    sig_img = Image.open(io.BytesIO(sig_bytes)).convert("L")
    sig_img = ImageOps.autocontrast(sig_img, cutoff=2)
    np_img = np.array(sig_img)
    
    # Paper threshold
    threshold = 175
    alpha = np.where(np_img < threshold, 255, 0).astype(np.uint8)
    
    h, w = np_img.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 10
    rgba[:, :, 1] = 10
    rgba[:, :, 2] = 10
    rgba[:, :, 3] = alpha
    
    clean_sig = Image.fromarray(rgba, mode="RGBA")
    bbox = clean_sig.getbbox()
    if bbox:
        clean_sig = clean_sig.crop(bbox)
        
    aspect = clean_sig.height / clean_sig.width
    sw = target_w
    sh = int(target_w * aspect)
    if sh > 75:
        sh = 75
        sw = int(sh / aspect)
        
    return clean_sig.resize((sw, sh), Image.Resampling.LANCZOS)

st.write("### 1️⃣ Target Photo (Gallery ya Camera)")
photo_mode = st.radio("Source Photo:", ["📁 Gallery Upload", "📷 Live Camera"], horizontal=True)

final_image_data = None
if photo_mode == "📁 Gallery Upload":
    uploaded_photo = st.file_uploader("Passport ke liye photo select karein", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_photo:
        final_image_data = uploaded_photo.read()
else:
    camera_photo = st.camera_input("Camera se click karein")
    if camera_photo:
        final_image_data = camera_photo.read()

st.write("---")
st.write("### 2️⃣ Signature Photo (Optional)")
uploaded_sign = st.file_uploader("Kagaz par kiye sign ki photo dalein (Agar lagana ho toh)", type=["jpg", "jpeg", "png", "webp"])

if final_image_data:
    if st.button("⚡ GENERATE STUDIO PASSPORT PHOTO NOW", type="primary", use_container_width=True):
        with st.spinner("Passport Photo Taiyaar Ho Rahi Hai..."):
            try:
                raw_img = Image.open(io.BytesIO(final_image_data)).convert("RGB")
                
                # 1. Natural Studio Face Balancing
                studio_face = natural_studio_lighting(raw_img)
                
                # 2. AI Background Removal (Free Local Micro Model)
                cutout_pil = remove(studio_face, session=session).convert("RGBA")
                
                # 3. Person ke actual bounds nikalna (Head aur Shoulders ka bbox)
                bbox = cutout_pil.getbbox()
                if bbox:
                    # Bounding box ke hisaab se crop taaki sir ke upar faltu jagah na bache
                    bx0, by0, bx1, by1 = bbox
                    person_w = bx1 - bx0
                    person_h = by1 - by0
                    
                    # Passport ratio target: 413 x 531
                    target_w, target_h = 413, 531
                    
                    # Ideal passport: Person height photo ka lagbhag 80-85% cover kare
                    crop_h = int(person_h / 0.82)
                    crop_w = int(crop_h * (target_w / target_h))
                    
                    # Center around the person horizontally
                    center_x = (bx0 + bx1) // 2
                    left = max(0, center_x - crop_w // 2)
                    top = max(0, by0 - int(person_h * 0.10)) # Sir ke upar standard 10% studio space
                    
                    # Agar image bounds se bahar jaye toh adjust karein
                    if left + crop_w > cutout_pil.width:
                        left = max(0, cutout_pil.width - crop_w)
                    if top + crop_h > cutout_pil.height:
                        crop_h = cutout_pil.height - top
                        crop_w = int(crop_h * (target_w / target_h))
                        
                    cropped_cutout = cutout_pil.crop((left, top, left + crop_w, top + crop_h))
                else:
                    cropped_cutout = cutout_pil
                    
                # Resize to standard passport dimensions (3.5cm x 4.5cm @ 300 DPI)
                cropped_cutout = cropped_cutout.resize((413, 531), Image.Resampling.LANCZOS)
                
                # 4. Solid Studio Sky-Blue Background (RGB: 135, 180, 235)
                bg = Image.new("RGBA", (413, 531), (135, 180, 235, 255))
                composed = Image.alpha_composite(bg, cropped_cutout).convert("RGB")
                
                # 5. Signature Overlay (Sirf tabhi aayega jab user upload karega)
                if uploaded_sign:
                    try:
                        clean_sig = process_signature(uploaded_sign.read(), target_w=190)
                        sig_x = int((413 - clean_sig.width) / 2)
                        sig_y = int(531 * 0.74) # Lower chest area
                        composed.paste(clean_sig, (sig_x, sig_y), clean_sig)
                    except Exception as sig_err:
                        st.warning(f"Signature process nahi ho paya: {sig_err}")
                
                # 6. Clean Thin Studio White Border
                draw = ImageDraw.Draw(composed)
                draw.rectangle([(0, 0), (412, 530)], outline="white", width=5)
                
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
            except Exception as ex:
                st.error(f"Error: {repr(ex)}")
