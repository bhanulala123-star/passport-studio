import streamlit as st
from PIL import Image
import io
import os
from google import genai
from google.genai import types

st.set_page_config(page_title="Studio Passport AI Maker", page_icon="📸", layout="centered")

# --- 🔒 PASSCODE ACCESS CONTROL ---
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
            st.error("Golat Code! Please sahi access code enter karein.")
    st.stop()
# ------------------------------

st.markdown("""
    <h2 style='text-align: center; color: #0d6efd;'>📸 Premium AI Studio Passport Photo Maker</h2>
    <p style='text-align: center; color: gray; font-size: 14px;'>
    Gemini-2.5-Flash Multimodal AI • Original Hair Safe • Pure Studio Blue BG • White Round Border • Signature
    </p>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("⚙️ AI Credentials")
    api_key_input = st.text_input("Gemini API Key (Provide Google Studio Key):", type="password")
    st.info("💡 Yeh app camera/file se photos lekar use modern Gemini Multimodal API se processing karti hai.")
    if st.button("Logout"):
        st.session_state.unlocked = False
        st.rerun()

def call_passport_generation_api(image_bytes, signature_name, api_key):
    """
    Direct multimodal pipeline using gemini-2.5-flash-image
    Focusing on natural face/hair restoration, studio sky-blue backdrop (#91b9eb), rounded border & sign.
    """
    # Initialize Google GenAI client safely
    client = genai.Client(api_key=api_key)
    
    # Load raw photo stream
    original_pil = Image.open(io.BytesIO(image_bytes))
    
    # Precise Image generation prompt with 2nd Image validation features
    model_prompt = f"""
    Strictly transform this uploaded user photo into a professional, vertical studio passport photo:
    
    1. FACE & IDENTITY PROTECTION (Strictest Priority): Fully retain the subject's 100% natural facial identity, exact age, gaze, and original hair strands (especially natural grey hairs) as in the input. Do NOT smudge, alter, crop, or artificialize any natural facial details. Avoid all cut-off or truncations of the hair.
    
    2. SOLID STUDIO BACKGROUND: Extract the person from the current background with seamless, feather-blended edge processing, placing them on an entirely solid studio sky-blue backdrop (#91b9eb) with balanced, soft studio back-fill lighting.
    
    3. OFFICIAL BORDERING: Framed centered from head to shoulder in standard vertical 3.5cm x 4.5cm passport crop (3:4 ratio) with a smooth, soft-rounded 4-pixel-wide white inner border outlining the image.
    
    4. SIGNATURE EMBEDDING: Seamlessly natural handwritten cursive signature '{signature_name}' overlayed in black ink across the lower chest area/collar, clearly legible, but ensuring it does not overlap with or obscure the chin or face.
    
    Return the finalized vertical passport portrait in rich 3:4 resolution without any distortions.
    """
    
    # Multimodal image API generation invoke
    generation_response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[original_pil, model_prompt],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio="3:4"
            )
        )
    )
    
    # Iterate and capture returned image buffers
    for part in generation_response.candidates[0].content.parts:
        if part.inline_data:
            return part.inline_data.data
            
    return None

photo_mode = st.radio("Sourse Photo Kahan Se Lena Hai?", ["📁 Gallery / File Upload", "📷 Live Camera Capture"], horizontal=True)

final_image_data = None
if photo_mode == "📁 Gallery / File Upload":
    uploaded_photo = st.file_uploader("1️⃣ Target Photo Upload Karein", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_photo:
        final_image_data = uploaded_photo.read()
else:
    camera_photo = st.camera_input("1️⃣ Camera Se Click Karein")
    if camera_photo:
        final_image_data = camera_photo.read()

sig_text = st.text_input("Signature Name (Jaise Pankaj Yadav):", value="Pankaj Yadav")

if final_image_data:
    if st.button("⚡ GENERATE STUDIO PASSPORT PHOTO NOW", type="primary", use_container_width=True):
        active_key = api_key_input.strip() or os.environ.get("GEMINI_API_KEY", "")
        
        if not active_key:
            st.error("⚠️ Sidebar kholein (>> upar left me) aur apni valid Gemini API Key paste kijiye.")
        else:
            with st.spinner("Multimodal Gemini-2.5-Flash-Image pipeline se quality studio portrait build ho raha hai..."):
                try:
                    result_passport_bytes = call_passport_generation_api(final_image_data, sig_text, active_key)
                    if result_passport_bytes:
                        st.success("✅ Studio Passport Photo Safaltapurvak Taiyaar Ho Gayi!")
                        st.image(result_passport_bytes, caption=f"AI Passport Image (3.5cm x 4.5cm)", width=280)

                        st.download_button(
                            label="📥 DOWNLOAD PASSPORT PHOTO",
                            data=result_passport_bytes,
                            file_name="studio_passport.jpg",
                            mime="image/jpeg",
                            use_container_width=True
                        )
                    else:
                        st.warning("⚠️ API response issue. Please parameters verification kijiye.")
                except Exception as ex:
                    st.error(f"Execution Error: {str(ex)}")
