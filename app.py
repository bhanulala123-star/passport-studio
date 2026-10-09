import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import google.generativeai as genai

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

DEFAULT_API_KEY = ""

st.markdown("""
    <h2 style='text-align: center; color: #0d6efd;'>📸 Studio Passport Photo Maker</h2>
    <p style='text-align: center; color: gray; font-size: 14px;'>
    Light Blue Studio BG • White Border • Face Glow • Jet-Black Sign
    </p>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("⚙️ Settings")
    api_key_input = st.text_input("Gemini API Key (Optional):", value=DEFAULT_API_KEY, type="password")
    if api_key_input:
        genai.configure(api_key=api_key_input)
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

def smart_passport_frame(img):
    h, w = img.shape[:2]
    # Agar photo pehle se hi lagbhag 3.5:4.5 ratio mein hai
    if 0.70 <= (w / h) <= 0.85:
        return cv2.resize(img, (413, 531), interpolation=cv2.INTER_AREA)

    target_ratio = 3.5 / 4.5
    current_ratio = w / h

    if current_ratio > target_ratio:
        # Photo chaudi (wide) hai, center se passport width crop karein
        new_w = int(h * target_ratio)
        x1 = max(0, (w - new_w) // 2)
        x2 = min(w, x1 + new_w)
        cropped = img[:, x1:x2]
    else:
        # Photo lambi hai, upar/chest region ko priority dekar crop karein
        new_h = int(w / target_ratio)
        y1 = int((h - new_h) * 0.25)
        y1 = max(0, y1)
        y2 = min(h, y1 + new_h)
        cropped = img[y1:y2, :]

    return cv2.resize(cropped, (413, 531), interpolation=cv2.INTER_AREA)

def apply_studio_blue_bg(img):
    h, w = img.shape[:2]
    studio_blue = np.array([235, 185, 145], dtype=np.uint8) # Studio light blue (BGR)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    rect = (8, 8, w - 16, h - 16)
    try:
        cv2.grabCut(img, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        m2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        m2 = cv2.GaussianBlur(m2.astype(np.float32), (7, 7), 0)
        m3 = np.repeat(m2[:, :, np.newaxis], 3, axis=2)
        bg = np.full((h, w, 3), studio_blue, dtype=np.uint8)
        return (img * m3 + bg * (1.0 - m3)).astype(np.uint8)
    except Exception:
        return img

def apply_face_glow(img):
    try:
        smooth = cv2.bilateralFilter(img, d=7, sigmaColor=45, sigmaSpace=45)
        lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        l_gora = cv2.convertScaleAbs(l_clahe, alpha=1.12, beta=14)
        glow_bgr = cv2.cvtColor(cv2.merge([l_gora, a, b]), cv2.COLOR_LAB2BGR)
        gaussian = cv2.GaussianBlur(glow_bgr, (0, 0), 1.5)
        return cv2.addWeighted(glow_bgr, 1.18, gaussian, -0.18, 0)
    except Exception:
        return img

def add_white_border(img):
    h, w = img.shape[:2]
    res = img.copy()
    margin = 12
    cv2.rectangle(res, (margin, margin), (w - margin, h - margin), (255, 255, 255), 4)
    return res

def process_signature(sig_img):
    if len(sig_img.shape) == 3:
        if sig_img.shape[2] == 4:
            b, g, r, a = cv2.split(sig_img)
            gray = cv2.cvtColor(cv2.merge([b, g, r]), cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(sig_img, cv2.COLOR_BGR2GRAY)
            a = None
    else:
        gray = sig_img.copy()
        a = None

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12)
    if a is not None:
        thresh = cv2.bitwise_and(thresh, a)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(thresh, kernel, iterations=1)
    black = np.zeros_like(gray)
    return cv2.merge([black, black, black, mask])

def overlay_signature(photo, sig_img):
    solid_sig = process_signature(sig_img)
    h, w = photo.shape[:2]
    sig_h = int(h * 0.12)
    sig_w = int(w * 0.60)
    resized_sig = cv2.resize(solid_sig, (sig_w, sig_h), interpolation=cv2.INTER_AREA)

    y1 = h - sig_h - 24
    y2 = y1 + sig_h
    x1 = (w - sig_w) // 2
    x2 = x1 + sig_w

    alpha = resized_sig[:, :, 3] / 255.0
    for c in range(3):
        photo[y1:y2, x1:x2, c] = alpha * resized_sig[:, :, c] + (1.0 - alpha) * photo[y1:y2, x1:x2, c]
    return photo

photo_mode = st.radio("Photo kaise select karni hai?", ["📁 Gallery / File Upload", "📷 Live Camera se Click Karein"], horizontal=True)

final_photo_data = None
if photo_mode == "📁 Gallery / File Upload":
    uploaded_photo = st.file_uploader("1️⃣ Photo Upload Karein", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_photo:
        final_photo_data = uploaded_photo.read()
else:
    camera_photo = st.camera_input("1️⃣ Camera ke samne dekh kar photo click karein")
    if camera_photo:
        final_photo_data = camera_photo.read()

sig_mode = st.radio("Signature kaise dena hai?", ["📁 Signature File Upload", "📷 Signature Ki Photo Kheenche"], horizontal=True)

final_sig_data = None
if sig_mode == "📁 Signature File Upload":
    uploaded_sig = st.file_uploader("2️⃣ Signature File Chuniye", type=["jpg", "jpeg", "png"])
    if uploaded_sig:
        final_sig_data = uploaded_sig.read()
else:
    camera_sig = st.camera_input("2️⃣ Kagaz par kiye sign ki photo kheenche")
    if camera_sig:
        final_sig_data = camera_sig.read()

if final_photo_data and final_sig_data:
    if st.button("⚡ GENERATE STUDIO PASSPORT PHOTO NOW", type="primary", use_container_width=True):
        with st.spinner("AI Studio Processing in progress..."):
            photo_bytes = np.asarray(bytearray(final_photo_data), dtype=np.uint8)
            img = cv2.imdecode(photo_bytes, cv2.IMREAD_COLOR)

            sig_bytes = np.asarray(bytearray(final_sig_data), dtype=np.uint8)
            sig = cv2.imdecode(sig_bytes, cv2.IMREAD_UNCHANGED)

            # Smart Passport Auto Framing
            framed = smart_passport_frame(img)

            # Blue Studio Background
            blue_bg = apply_studio_blue_bg(framed)

            # Face Glow & Skin Tone
            glowing = apply_face_glow(blue_bg)

            # Border
            bordered = add_white_border(glowing)

            # Jet Black Sign Overlay
            final_output = overlay_signature(bordered, sig)

            final_rgb = cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(final_rgb)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=95)
            byte_im = buf.getvalue()

            st.success("✅ Studio Passport Photo Taiyaar Ho Gayi!")
            st.image(final_rgb, caption="Passport Preview (3.5cm x 4.5cm)", width=260)

            st.download_button(
                label="📥 DOWNLOAD PASSPORT PHOTO",
                data=byte_im,
                file_name="studio_passport.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
