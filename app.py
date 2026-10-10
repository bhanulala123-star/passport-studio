import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps
import io
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
    100% Free • Perfect Face Centering • Studio Soft BG • Signature Overlay
    </p>
""", unsafe_allow_html=True)

@st.cache_resource
def load_session():
    return new_session("silueta")

session = load_session()

def create_studio_gradient_bg(w, h):
    """Soft spotlight sky-blue studio background using pure Pillow/Numpy"""
    y_coords, x_coords = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h * 0.38
    dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2) / (w * 0.8)
    factor = np.clip(dist, 0.0, 1.0)
    
    # Bright studio sky-blue center (165, 202, 245) to edge (125, 168, 225)
    r = (165 * (1 - factor) + 125 * factor).astype(np.uint8)
    g = (202 * (1 - factor) + 168 * factor).astype(np.uint8)
    b = (245 * (1 - factor) + 225 * factor).astype(np.uint8)
    
    arr = np.dstack((r, g, b, np.full((h, w), 255, dtype=np.uint8)))
    return Image.fromarray(arr, mode="RGBA")

def perfect_passport_centering(cutout_img, target_w=413, target_h=531):
    """Person ko detect karke exact photo studio jaisa centered passport crop karna"""
    bbox = cutout_img.getbbox()
    if not bbox:
        return cutout_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    bx0, by0, bx1, by1 = bbox
    person_w = bx1 - bx0
    person_h = by1 - by0
    
    aspect = target_w / target_h
    
    # Passport standard: Sir se chhati (chest) tak ka hissa photo ka 78% cover kare
    crop_h = int(person_h * 0.90)
    crop_w = int(crop_h * aspect)
    
    # Center horizontally on the person
    person_cx = (bx0 + bx1) // 2
    left = person_cx - (crop_w // 2)
    # Head ke upar standard 10% studio gap
    top = by0 - int(person_h * 0.08)
    
    # Agar boundary se bahar jaye toh adjust karein
    if crop_w > cutout_img.width:
        crop_w = cutout_img.width
        crop_h = int(crop_w / aspect)
        left = 0
        top = by0 - int(person_h * 0.05)
    else:
        left = max(0, min(left, cutout_img.width - crop_w))
        top = max(0, min(top, cutout_img.height - crop_h))
        
    cropped = cutout_img.crop((left, top, left + crop_w, top + crop_h))
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
        with st.spinner("AI photo process karke perfect studio framing me convert kar raha hai..."):
            try:
                raw_img = Image.open(uploaded_photo).convert("RGB")
                
                # 1. Natural Studio Color & Light Polish
                bright = ImageEnhance.Brightness(raw_img).enhance(1.06)
                contrast = ImageEnhance.Contrast(bright).enhance(1.05)
                sharp = ImageEnhance.Sharpness(contrast).enhance(1.20)
                
                # 2. Free AI Background Removal
                cutout_pil = remove(sharp, session=session).convert("RGBA")
                
                # 3. Smart Centering & Passport Framing (Side khaalipan khatam)
                person_centered = perfect_passport_centering(cutout_pil, 413, 531)
                
                # 4. Studio Soft Gradient Background (413 x 531)
                bg = create_studio_gradient_bg(413, 531)
                composed = Image.alpha_composite(bg, person_centered).convert("RGB")
                
                # 5. Signature Overlay
                draw = ImageDraw.Draw(composed)
                if sig_mode == "✏️ Typed Cursive Name" and typed_sig.strip():
                    try:
                        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
                    except Exception:
                        font = ImageFont.load_default()
                    draw.text((115, 410), typed_sig.strip(), fill=(20, 20, 20), font=font)
                    
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
            except Exception as ex:
                st.error(f"Error: {repr(ex)}")
