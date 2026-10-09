import streamlit as st
from PIL import Image
import io
import os
from google import genai
from google.genai import types

st.set_page_config(page_title="Studio Passport Maker AI", page_icon="📸", layout="centered")

# --- 🔒 PASSWORD PROTECTION ---
APP_PASSWORD = "1234"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.subheader("🔒 Studio Login")
    entered_pw = st.text_input("Enter Passcode:", type="password")
    if st.button("Unlock App", type="primary", use_container_width=True):
        if entered_pw == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Galat Password!")
    st.stop()
# ------------------------------

st.markdown("""
    <h2 style='text-align: center; color: #0d6efd;'>📸 Studio Passport Photo Maker</h2>
    <p style='text-align: center; color: gray; font-size: 14px;'>
    Gemini Native AI Studio Engine • Natural Hair • Light Blue Studio BG • Signature Overlay
    </p>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("⚙️ Settings")
    api_key_input = st.text_input("Gemini API Key:", type="password")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

def generate_studio_passport_via_gemini_api(photo_bytes, signature_name, api_key):
    # Initialize the modern official Google GenAI Client
    client = genai.Client(api_key=api_key)
    
    pil_photo = Image.open(io.BytesIO(photo_bytes))
    
    # Exact Prompt for Studio Quality Match
    prompt = f"""
    Transform this uploaded person's portrait into an authentic, professional, vertical studio passport photo:
    1. KEEP THE PERSON'S IDENTITY 100% INTACT: Retain all natural facial features, skin texture, wrinkles, eyes, and natural grey hair strands exactly as in the photo. Do not cut or crop the hair or top of the head.
    2. SOLID LIGHT-BLUE STUDIO BACKGROUND: Completely isolate the person with clean, soft-lit edges against a solid, studio light-blue / sky-blue background.
    3. PASSPORT FRAMING: Professional bust framing from head to shoulders in 3.5cm x 4.5cm vertical passport aspect ratio. Include a subtle, clean white border with slightly rounded corners around the image.
    4. NATURAL SIGNATURE OVERLAY: Add an elegant, realistic black cursive handwritten signature '{signature_name}' overlaying cleanly on top of the lower chest/clothing area, readable and studio-aligned.
    5. NATURAL STUDIO LIGHTING: Soft, balanced studio portrait lighting on the face with no harsh shadows.
    """
    
    # Call Gemini Image Model
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[pil_photo, prompt],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio="3:4"
            )
        )
    )
    
    # Extract returned image bytes
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            return part.inline_data.data
            
    return None

photo_mode = st.radio("Photo kaise select karni hai?", ["📁 Gallery / File Upload", "📷 Live Camera se Click Karein"], horizontal=True)

final_photo_data = None
if photo_mode == "📁 Gallery / File Upload":
    uploaded_photo = st.file_uploader("1️⃣ Photo Upload Karein", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_photo:
        final_photo_data = uploaded_photo.read()
else:
    camera_photo = st.camera_input("1️⃣ Live Camera se photo lein")
    if camera_photo:
        final_photo_data = camera_photo.read()

sig_text = st.text_input("Signature Name:", value="Pankaj Yadav")

if final_photo_data:
    if st.button("⚡ GENERATE STUDIO PASSPORT PHOTO NOW", type="primary", use_container_width=True):
        active_key = api_key_input.strip() or os.environ.get("GEMINI_API_KEY", "")
        
        if not active_key:
            st.error("⚠️ Sidebar me apni valid Gemini API Key paste kijiye.")
        else:
            with st.spinner("Gemini AI Studio Model se authentic passport photo generate ho rahi hai..."):
                try:
                    result_img_bytes = generate_studio_passport_via_gemini_api(final_photo_data, sig_text, active_key)
                    if result_img_bytes:
                        st.success("✅ Studio Passport Photo Taiyaar Ho Gayi!")
                        st.image(result_img_bytes, caption="AI Studio Portrait (3.5cm x 4.5cm)", width=280)

                        st.download_button(
                            label="📥 DOWNLOAD PASSPORT PHOTO",
                            data=result_img_bytes,
                            file_name="studio_passport.jpg",
                            mime="image/jpeg",
                            use_container_width=True
                        )
                    else:
                        st.error("Model se image generate nahi ho payi. Dobara try karein.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
