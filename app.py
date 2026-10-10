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
    <h2 style='text-align: center; color: #0d6efd;'>📸 Studio Passport Photo Maker</h2>
    <p style='text-align: center; color: gray; font-size: 14px;'>
    Studio Lighting Touch-up • Sign Image Auto-Extract • Clean Solid BG
    </p>
""", unsafe_allow_html=True)

@st.cache_resource
def load_lightweight_model():
    return new_session("silueta")

session = load_lightweight_model()

def enhance_portrait_studio(pil_img):
    """Studio style brightness, contrast aur sharpness touchup"""
    # 1. Exposure / Brightness Boost
    bright_enhancer = ImageEnhance.Brightness(pil_img)
    img_bright = bright_enhancer.enhance(1.08)

    # 2. Studio Contrast
    contrast_enhancer = ImageEnhance.Contrast(img_bright)
    img_contrast = contrast_enhancer.enhance(1.10)

    # 3. Clarity & Sharpness (Chehre aur aankhon ki detail ke liye)
    sharp_enhancer = ImageEnhance.Sharpness(img_contrast)
    img_sharp = sharp_enhancer.enhance(1.25)
    
    # 4. Color Warmth & Vibrancy
    color_enhancer = ImageEnhance.Color(img_sharp)
    img_final = color_enhancer.enhance(1.05)
    return img_final

def process_signature_photo(sig_bytes, target_width=220):
    """Signature photo me se white kagaz background hata kar sirf ink sign nikalna"""
    sig_img = Image.open(io.BytesIO(sig_bytes)).convert("L") # Grayscale
    
    # Auto-contrast to separate ink from paper
    sig_img = ImageOps.autocontrast(sig_img, cutoff=2)
    
    # Thresholding: Ink ko black aur paper ko transparent
    np_img = np.array(sig_img)
    # Paper threshold
    threshold = 175
    
    # Create RGBA
    alpha = np.where(np_img < threshold, 255, 0).astype(np.uint8)
    
    # Inverted ink: Black ink color (RGB: 15, 15, 15)
    h, w = np_img.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 20  # R
    rgba[:, :, 1] = 20  # G
    rgba[:, :, 2] = 20  # B
    rgba[:, :, 3] = alpha  # Transparency
    
    clean_sig = Image.fromarray(rgba, mode="RGBA")
    
    # Crop empty transparent borders
    bbox = clean_sig.getbbox()
    if bbox:
        clean_sig = clean_sig.crop(bbox)
        
    # Resize signature proportional to passport size
    aspect = clean_sig.height / clean_sig.width
    sig_w = target_width
    sig_h = int(target_width * aspect)
    if sig_h > 90:
        sig_h = 90
        sig_w = int(sig_h / aspect)
        
    return clean_sig.resize((sig_w, sig_h), Image.Resampling.LANCZOS)

st.write("### 1️⃣ Target Photo (Chehra)")
photo_mode = st.radio("Source Photo:", ["📁 Gallery Upload", "📷 Live Camera"], horizontal=True)

final_image_data = None
if photo_mode == "📁 Gallery Upload":
    uploaded_photo = st.file_uploader("Chehre ki photo select karein", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_photo:
        final_image_data = uploaded_photo.read()
else:
    camera_photo = st.camera_input("Camera se photo click karein")
    if camera_photo:
        final_image_data = camera_photo.read()

st.write("---")
st.write("### 2️⃣ Signature Photo (Kagaz Par Kiya Hua Sign)")
sig_file = st.file_uploader("Kagaz par sign ki hui photo upload karein (Optional)", type=["jpg", "jpeg", "png", "webp"])

if final_image_data:
    if st.button("⚡ GENERATE STUDIO PASSPORT PHOTO NOW", type="primary", use_container_width=True):
        with st.spinner("Processing: Background Removal + Face Studio Lighting + Signature Overlay..."):
            try:
                # 1. AI Cutout via local micro model
                input_pil = Image.open(io.BytesIO(final_image_data)).convert("RGB")
                
                # Face Lighting & Clarity Boost
                studio_face = enhance_portrait_studio(input_pil)
                
                # Background Cutout
                cutout_pil = remove(studio_face, session=session).convert("RGBA")

                # 2. Studio Sky-Blue BG
                bg = Image.new("RGBA", cutout_pil.size, (145, 185, 235, 255))
                composed = Image.alpha_composite(bg, cutout_pil).convert("RGB")

                # 3. 3.5cm x 4.5cm Passport Ratio Crop (413 x 531)
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
                    top = int((h - new_h) * 0.10)
                    composed = composed.crop((0, top, w, top + new_h))

                composed = composed.resize((target_w, target_h), Image.Resampling.LANCZOS)

                # 4. Signature Photo Auto Overlay (Kagaz ka background gayab karke)
                if sig_file:
                    try:
                        clean_sig = process_signature_photo(sig_file.read(), target_width=180)
                        # Signature position: lower chest area / bottom
                        sig_x = int((target_w - clean_sig.width) / 2)
                        sig_y = int(target_h * 0.74)
                        composed.paste(clean_sig, (sig_x, sig_y), clean_sig)
                    except Exception as sig_err:
                        st.warning(f"Signature process nahi ho paya: {sig_err}")

                # 5. White Rounded Studio Border
                draw = ImageDraw.Draw(composed)
                draw.rounded_rectangle([(8, 8), (target_w - 8, target_h - 8)], radius=12, outline="white", width=4)

                buf = io.BytesIO()
                composed.save(buf, format="JPEG", quality=95)
                out_bytes = buf.getvalue()

                st.success("✅ Studio Passport Photo Taiyaar!")
                st.image(out_bytes, caption="Studio Passport Preview (3.5cm x 4.5cm)", width=280)

                st.download_button(
                    label="📥 DOWNLOAD PASSPORT PHOTO",
                    data=out_bytes,
                    file_name="studio_passport.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
            except Exception as ex:
                st.error(f"Error: {repr(ex)}")
