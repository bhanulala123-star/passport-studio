import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
from rembg import remove

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
    <h2 style='text-align: center; color: #0d6efd;'>📸 AI Studio Passport Photo Maker</h2>
    <p style='text-align: center; color: gray; font-size: 14px;'>
    Zero API Required • Natural Hair Preservation • Solid Studio Blue BG • Signature Overlay
    </p>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("⚙️ System Status")
    st.success("✅ On-Device AI Active (No API Key Needed)")
    if st.button("Logout"):
        st.session_state.unlocked = False
        st.rerun()

def process_passport_photo(img_bytes, signature_text):
    original_img = Image.open(io.BytesIO(img_bytes))
    
    # 1. AI Cutout with hair protection (Runs locally via rembg AI model)
    cutout = remove(original_img)
    
    # 2. Studio Sky-Blue Background (RGB: 145, 185, 235)
    bg = Image.new("RGBA", cutout.size, (145, 185, 235, 255))
    composed = Image.alpha_composite(bg, cutout).convert("RGB")
    
    # 3. Standard Passport Ratio Crop & Resize (3.5cm x 4.5cm -> 413 x 531)
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
        top = int((h - new_h) * 0.15)
        composed = composed.crop((0, top, w, top + new_h))
        
    composed = composed.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    # 4. White Rounded Border
    draw = ImageDraw.Draw(composed)
    draw.rounded_rectangle([(8, 8), (target_w - 8, target_h - 8)], radius=14, outline="white", width=4)
    
    # 5. Cursive / Elegant Signature Overlay on collar area
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
        
    if signature_text.strip():
        draw.text((int(target_w * 0.26), int(target_h * 0.76)), signature_text.strip(), fill=(15, 15, 15), font=font)
        
    return composed

photo_mode = st.radio("Source Photo Kahan Se Lena Hai?", ["📁 Gallery / File Upload", "📷 Live Camera Capture"], horizontal=True)

final_image_data = None
if photo_mode == "📁 Gallery / File Upload":
    uploaded_photo = st.file_uploader("1️⃣ Target Photo Upload Karein", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_photo:
        final_image_data = uploaded_photo.read()
else:
    camera_photo = st.camera_input("1️⃣ Camera Se Click Karein")
    if camera_photo:
        final_image_data = camera_photo.read()

sig_text = st.text_input("Signature Name:", value="Pankaj Yadav")

if final_image_data:
    if st.button("⚡ GENERATE STUDIO PASSPORT PHOTO NOW", type="primary", use_container_width=True):
        with st.spinner("AI photo process kar raha hai (Hair Safe & Studio BG)..."):
            try:
                res_img = process_passport_photo(final_image_data, sig_text)
                
                buf = io.BytesIO()
                res_img.save(buf, format="JPEG", quality=95)
                output_bytes = buf.getvalue()
                
                st.success("✅ Studio Passport Photo Taiyaar Ho Gayi!")
                st.image(output_bytes, caption="AI Studio Portrait (3.5cm x 4.5cm)", width=280)

                st.download_button(
                    label="📥 DOWNLOAD PASSPORT PHOTO",
                    data=output_bytes,
                    file_name="studio_passport.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
            except Exception as ex:
                st.error(f"Error: {str(ex)}")
