import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps, ImageFilter
import io
import urllib.request
import numpy as np
from rembg import remove, new_session

st.set_page_config(page_title="Professional Passport Maker", page_icon="📸", layout="centered")

# --- 🔒 PASSCODE ---
ACCESS_CODE = "1234"
if "unlocked" not in st.session_state:
    st.session_state.unlocked = False

if not st.session_state.unlocked:
    st.subheader("🔒 Secure Entry")
    entered_code = st.text_input("Enter Passcode:", type="password")
    if st.button("Unlock App", type="primary", use_container_width=True):
        if entered_code == ACCESS_CODE:
            st.session_state.unlocked = True
            st.rerun()
        else:
            st.error("Wrong passcode!")
    st.stop()

st.markdown("""
    <h2 style='text-align: center; color: #0d6efd;'>📸 Passport Photo & Sign Maker</h2>
    <p style='text-align: center; color: gray; font-size: 14px;'>
    Auto-Crop • Studio Lighting • White Border • Perfect Signature Overlay
    </p>
""", unsafe_allow_html=True)

# 1. Load Background Removal Model Fast
@st.cache_resource
def load_session():
    return new_session("silueta")
session = load_session()

# 2. Load Stylish Cursive Font for Signature
@st.cache_resource
def get_signature_font():
    font_path = "cursive_font.ttf"
    try:
        # AlexBrush gives a beautiful natural cursive look
        url = "https://github.com/google/fonts/raw/main/ofl/alexbrush/AlexBrush-Regular.ttf"
        urllib.request.urlretrieve(url, font_path)
        return ImageFont.truetype(font_path, 48) # Font size for signature
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
        except Exception:
            return ImageFont.load_default()
cursive_font = get_signature_font()

# 3. Studio Lighting (Face Glow)
def apply_studio_glow(pil_img):
    """Brightens face and adds crisp studio quality"""
    arr = np.array(pil_img).astype(np.float32) / 255.0
    arr = np.power(arr, 0.82) # Lift shadows
    arr = np.clip((arr - 0.5) * 1.08 + 0.52, 0.0, 1.0) # Contrast
    base_img = Image.fromarray((arr * 255).astype(np.uint8))
    
    sharp = base_img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=130, threshold=3))
    return ImageEnhance.Color(sharp).enhance(1.05)

# 4. Perfect Sky-Blue Background
def create_studio_bg(w, h):
    y_coords, x_coords = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h * 0.4
    dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2) / (w * 0.9)
    factor = np.clip(dist, 0.0, 1.0)
    
    r = (145 * (1 - factor) + 110 * factor).astype(np.uint8)
    g = (195 * (1 - factor) + 160 * factor).astype(np.uint8)
    b = (245 * (1 - factor) + 230 * factor).astype(np.uint8)
    
    arr = np.dstack((r, g, b, np.full((h, w), 255, dtype=np.uint8)))
    return Image.fromarray(arr, mode="RGBA")

# 5. Exact Passport Size Crop (Chest Up)
def exact_passport_crop(cutout_img, target_w=413, target_h=531):
    """Ensure person is centered and cropped to exact 3.5x4.5 ratio"""
    bbox = cutout_img.getbbox()
    if not bbox:
        return cutout_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    bx0, by0, bx1, by1 = bbox
    person_w = bx1 - bx0
    person_h = by1 - by0
    aspect = target_w / target_h
    
    # Force crop box to passport aspect ratio
    crop_h = int(person_h * 0.85) # Frame chest up
    crop_w = int(crop_h * aspect)
    
    person_cx = (bx0 + bx1) // 2
    left = person_cx - (crop_w // 2)
    top = by0 - int(person_h * 0.06) # Space above head
    
    # Check boundaries
    if crop_w > cutout_img.width:
        crop_w = cutout_img.width
        crop_h = int(crop_w / aspect)
        left = 0
        top = by0 - int(person_h * 0.03)
    else:
        left = max(0, min(left, cutout_img.width - crop_w))
        top = max(0, min(top, cutout_img.height - crop_h))
        
    cropped = cutout_img.crop((left, top, left + crop_w, top + crop_h))
    # Final resize to standard pixel dimensions (413x531)
    return cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)

st.write("### 1️⃣ Target Photo Upload")
uploaded_photo = st.file_uploader("Select Photo for Passport:", type=["jpg", "jpeg", "png", "webp"])

st.write("### 2️⃣ Signature Settings")
col1, col2 = st.columns(2)
with col1:
    sig_mode = st.radio("Signature Style:", ["✏️ Typed Cursive", "📷 Upload Signature Photo", "❌ No Signature"], index=0)
with col2:
    if sig_mode == "✏️ Typed Cursive":
        typed_sig = st.text_input("Enter Name:", value="Pankaj Yadav")
    elif sig_mode == "📷 Upload Signature Photo":
        sig_file = st.file_uploader("Upload Paper Signature", type=["jpg", "png", "jpeg"])

if uploaded_photo:
    if st.button("⚡ GENERATE FINAL PASSPORT PHOTO", type="primary", use_container_width=True):
        with st.spinner("Processing image to exact passport standards..."):
            try:
                raw_img = Image.open(uploaded_photo).convert("RGB")
                
                # A) Face Glow & Enhancement
                glowing_face = apply_studio_glow(raw_img)
                
                # B) Remove Background
                cutout_pil = remove(glowing_face, session=session).convert("RGBA")
                
                # C) Exact Passport Size Crop (Crucial step for shape)
                # Standard size: 413x531 pixels (approx 3.5cm x 4.5cm at 300dpi)
                final_w, final_h = 413, 531
                person_cropped = exact_passport_crop(cutout_pil, final_w, final_h)
                
                # D) Add Blue Background
                bg = create_studio_bg(final_w, final_h)
                composed = Image.alpha_composite(bg, person_cropped).convert("RGB")
                
                draw = ImageDraw.Draw(composed)
                
                # E) Add Signature Overlay correctly
                if sig_mode == "✏️ Typed Cursive" and typed_sig.strip():
                    # Center the text at the bottom chest area
                    bbox = draw.textbbox((0, 0), typed_sig.strip(), font=cursive_font)
                    text_w = bbox[2] - bbox[0]
                    text_x = (final_w - text_w) // 2
                    text_y = int(final_h * 0.80) # Placing nicely on the chest
                    draw.text((text_x, text_y), typed_sig.strip(), fill=(15, 15, 15), font=cursive_font)
                    
                elif sig_mode == "📷 Upload Signature Photo" and sig_file:
                    sig_img = Image.open(sig_file).convert("L")
                    sig_img = ImageOps.autocontrast(sig_img, cutoff=2)
                    np_sig = np.array(sig_img)
                    alpha = np.where(np_sig < 175, 255, 0).astype(np.uint8)
                    rgba = np.zeros((np_sig.shape[0], np_sig.shape[1], 4), dtype=np.uint8)
                    rgba[:, :, :3] = 15 # Dark ink color
                    rgba[:, :, 3] = alpha
                    clean_sig = Image.fromarray(rgba, mode="RGBA")
                    
                    bbox = clean_sig.getbbox()
                    if bbox:
                        clean_sig = clean_sig.crop(bbox)
                        
                    # Resize signature properly to fit inside photo width
                    sig_target_w = 200
                    sig_target_h = int(sig_target_w * clean_sig.height / clean_sig.width)
                    clean_sig = clean_sig.resize((sig_target_w, sig_target_h), Image.Resampling.LANCZOS)
                    
                    sig_x = (final_w - clean_sig.width) // 2
                    sig_y = int(final_h * 0.78)
                    composed.paste(clean_sig, (sig_x, sig_y), clean_sig)

                # F) Add White Rounded Inner Border (Crucial for the classic look)
                draw.rounded_rectangle([(8, 8), (final_w - 8, final_h - 8)], radius=12, outline="white", width=4)
                
                # Output Preparation
                buf = io.BytesIO()
                composed.save(buf, format="JPEG", quality=98)
                out_bytes = buf.getvalue()
                
                st.success("✅ Studio Passport Photo Generated Successfully!")
                
                # Display with proper width so it looks like a passport
                st.image(out_bytes, caption="Final Passport Preview (3.5cm x 4.5cm)", width=320)
                
                st.download_button(
                    label="📥 DOWNLOAD PASSPORT PHOTO",
                    data=out_bytes,
                    file_name="professional_passport.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
            except Exception as ex:
                st.error(f"System Error: {repr(ex)}")
