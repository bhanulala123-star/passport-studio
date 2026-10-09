import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import google.generativeai as genai

st.set_page_config(page_title="Studio Passport Maker AI", page_icon="📸", layout="centered")

DEFAULT_API_KEY = ""

st.markdown("""
    <h2 style='text-align: center; color: #0d6efd;'>📸 Studio Passport Photo Maker</h2>
    <p style='text-align: center; color: gray; font-size: 14px;'>
    Light Blue Studio BG • White Border • Face Glow • Jet-Black Sign
    </p>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("⚙️ AI Settings")
    api_key_input = st.text_input("Gemini API Key:", value=DEFAULT_API_KEY, type="password")
    if api_key_input:
        genai.configure(api_key=api_key_input)

def detect_face(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    if len(faces) > 0:
        return max(faces, key=lambda b: b[2] * b[3])
    return (int(w * 0.22), int(h * 0.15), int(w * 0.55), int(h * 0.45))

def is_already_passport(h, w, box):
    bw = box[2]
    return (0.70 <= (w / h) <= 0.85) and (0.35 <= (bw / w) <= 0.65)

def smart_crop(img, box):
    h, w = img.shape[:2]
    x, y, bw, bh = box
    cx, cy = x + bw // 2, y + bh // 2
    crop_w = int(bw * 2.35)
    crop_h = int(crop_w * (4.5 / 3.5))
    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - int(bh * 1.25))
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)
    if y2 - y1 < crop_h: y1 = max(0, y2 - crop_h)
    if x2 - x1 < crop_w: x1 = max(0, x2 - crop_w)
    cropped = img[y1:y2, x1:x2]
    return cv2.resize(cropped, (413, 531), interpolation=cv2.INTER_AREA)

def apply_studio_blue_bg(img):
    h, w = img.shape[:2]
    studio_blue = np.array([235, 185, 145], dtype=np.uint8)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    rect = (10, 10, w - 20, h - 20)
    try:
        cv2.grabCut(img, mask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
        m2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        m2 = cv2.GaussianBlur(m2.astype(np.float32), (7, 7), 0)
        m3 = np.repeat(m2[:, :, np.newaxis], 3, axis=2)
        bg = np.full((h, w, 3), studio_blue, dtype=np.uint8)
        return (img * m3 + bg * (1.0 - m3)).astype(np.uint8)
    except Exception:
        return img

def apply_face_glow(img):
    smooth = cv2.bilateralFilter(img, d=7, sigmaColor=45, sigmaSpace=45)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    l_gora = cv2.convertScaleAbs(l_clahe, alpha=1.14, beta=16)
    glow_bgr = cv2.cvtColor(cv2.merge([l_gora, a, b]), cv2.COLOR_LAB2BGR)
    gaussian = cv2.GaussianBlur(glow_bgr, (0, 0), 1.8)
    return cv2.addWeighted(glow_bgr, 1.20, gaussian, -0.20, 0)

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

    y1 = h - sig_h - 22
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

            h, w = img.shape[:2]
            box = detect_face(img)

            if is_already_passport(h, w, box):
                framed = cv2.resize(img, (413, 531), interpolation=cv2.INTER_AREA)
            else:
                framed = smart_crop(img, box)

            blue_bg = apply_studio_blue_bg(framed)
            glowing = apply_face_glow(blue_bg)
            bordered = add_white_border(glowing)
            final_output = overlay_signature(bordered, sig)

            final_rgb = cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(final_rgb)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=95)
            byte_im = buf.getvalue()

            st.success("✅ Studio Passport Photo Taiyaar Ho Gayi!")
            st.image(final_rgb, caption="Preview", width=260)

            st.download_button(
                label="📥 DOWNLOAD PASSPORT PHOTO",
                data=byte_im,
                file_name="studio_passport.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
