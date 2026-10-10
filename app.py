import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import requests

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
    st.subheader("⚙️ Hugging Face Setup")
    user_hf_token = st.text_input("Paste Full HuggingFace Token (hf_...):", type="password")
    if st.button("Logout"):
        st.session_state.unlocked = False
        st.rerun()

def remove_background_free_ai(image_bytes, token):
    url = "https://router.huggingface.co/hf-inference/models/briaai/RMBG-1.4"
    headers = {
        "Authorization": f"Bearer {token.strip()}",
        "Content-Type": "application/octet-stream"
    }
    
    response = requests.post(url, headers=headers, data=image_bytes, timeout=60)
    
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content)).convert("RGBA")
    elif response.status_code == 503:
        st.warning("⏳ AI Model load ho raha hai (Warm up). Kripya 20 second baad dobara generate button dabayein.")
        return None
    else:
        st.error(f"Hugging Face Response ({response.status_code}): {response.text}")
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
        if not user_hf_token.strip():
            st.error("⚠️ Pehle Sidebar kholein (>> upar left me) aur poori HuggingFace token paste karein.")
        else:
            with st.spinner("Free Cloud AI photo process kar raha hai (Hair Safe & Studio BG)..."):
                try:
                    cutout_img = remove_background_free_ai(final_image_data, user_hf_token)
                    
                    if cutout_img is not None:
                        # 1. Solid Studio Sky-Blue BG (#91b9eb -> RGB 145, 185, 
