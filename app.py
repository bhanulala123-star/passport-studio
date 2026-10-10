import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import io

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
    Zero RAM Crash • Studio Sky-Blue Background • White Border • Signature Overlay
    </p>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("⚙️ System Status")
    st.success("✅ Engine Active (Lightweight & Stable)")
    if st.button("Logout"):
        st.session_state.unlocked = False
        st.rerun()

def smart_studio_passport(pil_img, signature_text):
    # 1. Standard Passport Ratio Crop (3.5cm x 4.5cm -> 413 x 531)
    target_w, target_h = 413, 531
    w, h = pil_img.size
    target_ratio = target_w / target_h
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        cropped = pil_img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = int((h - new_h) * 0.12)
        cropped = pil_img.crop((0, top, w, top + new_h))

    cropped = cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)
    img_np = np.array(cropped)

    # 2. Studio Sky-Blue Replacement with Soft Feathering
    # Studio Sky Blue: RGB (145, 185, 235)
    blue_bg = np.full_like(img_np, [145, 185, 235])

    # Smart edge mask creation (Head & Body protected)
    mask = Image.new("L", (target_w, target_h), 0)
    draw_mask = ImageDraw.Draw(mask)
    
    # Head & hair ellipse
    draw_mask.ellipse([(int(target_w * 0.10), int(target_h * 0.04)), 
                       (int(target_w * 0.90), int(target_h * 0.62))], fill=255)
    # Torso rectangle
    draw_mask.rectangle([(int(target_w * 0.05), int(target_h * 0.45)), 
                         (int(target_w * 0.95), target_h)], fill=255)
    
    # Soft feathering blur to avoid hard cut circles
    mask_blurred = mask.filter(ImageFilter.GaussianBlur(radius=18))
    mask_np = np.array(mask_blurred)[:, :, np.newaxis] / 255.0

    # Composite subject with Studio Blue Background
    composed_np = (img_np * mask_np + blue_bg * (1.0 - mask_np)).astype(np.uint8)
    final_pil = Image.fromarray(composed_np)

    # 3. Crisp White Rounded Border
    draw = ImageDraw.Draw(final_pil)
    draw.rounded_rectangle([(8, 8), (target_w - 8, target_h - 8)], radius=12, outline="white", width=4)

    # 4. Signature Overlay
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    if signature_text.strip():
        draw.text((int(target_w * 0.26), int(target_h * 0.76)), signature_text.strip(), fill=(20, 20, 20), font=font)

    return final_pil

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
        with st.spinner("Passport photo process ho rahi hai..."):
            try:
                pil_in = Image.open(io.BytesIO(final_image_data)).convert("RGB")
                res_img = smart_studio_passport(pil_in, sig_text)

                buf = io.BytesIO()
                res_img.save(buf, format="JPEG", quality=95)
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
                st.error(f"Error: {str(ex)}")
