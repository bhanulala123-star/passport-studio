import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
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
    100% Free Cloud AI • Natural Hair Safe • Solid Sky-Blue BG • Signature
    </p>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("⚙️ Free AI Setup")
    hf_token = st.text_input("Paste HuggingFace Token (hf_...):", type="password")
    if st.button("Logout"):
        st.session_state.unlocked = False
        st.rerun()

def remove_background_free_ai(image_bytes, token):
    API_URL = "https://api-inference.huggingface.co/models/briaai/RMBG-1.4"
    headers = {"Authorization": f"Bearer {token.strip()}"}
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    if response.status_code == 200:
        return response.content
    else:
        st.error(f"AI Server Error ({response.status_code}): {response.text}")
        return None

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
        if not hf_token:
            st.error("⚠️ Sidebar kholein (>> upar left me) aur apni HuggingFace token paste karein.")
        else:
            with st.spinner("Free Cloud AI photo process kar raha hai..."):
                try:
                    cutout_bytes = remove_background_free_ai(final_image_data, hf_token)
                    if cutout_bytes:
                        cutout_img = Image.open(io.BytesIO(cutout_bytes)).convert("RGBA")

                        # 1. Solid Studio Sky-Blue BG
                        bg = Image.new("RGBA", cutout_img.size, (145, 185, 235, 255))
                        composed = Image.alpha_composite(bg, cutout_img).convert("RGB")

                        # 2. 3.5cm x 4.5cm Passport Crop
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

                        # 3. White Rounded Border
                        draw = ImageDraw.Draw(composed)
                        draw.rounded_rectangle([(8, 8), (target_w - 8, target_h - 8)], radius=12, outline="white", width=4)

                        # 4. Signature
                        try:
                            font = ImageFont.truetype("DejaVuSans.ttf", 24)
                        except Exception:
                            font = ImageFont.load_default()

                        if sig_text.strip():
                            draw.text((int(target_w * 0.26), int(target_h * 0.76)), sig_text.strip(), fill=(15, 15, 15), font=font)

                        buf = io.BytesIO()
                        composed.save(buf, format="JPEG", quality=95)
                        out_bytes = buf.getvalue()

                        st.success("✅ Studio Passport Photo Taiyaar!")
                        st.image(out_bytes, caption="Studio Passport (3.5cm x 4.5cm)", width=280)

                        st.download_button(
                            label="📥 DOWNLOAD PASSPORT PHOTO",
                            data=out_bytes,
                            file_name="studio_passport.jpg",
                            mime="image/jpeg",
                            use_container_width=True
                        )
                except Exception as ex:
                    st.error(f"Error: {str(ex)}")
