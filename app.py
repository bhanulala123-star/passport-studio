import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

st.set_page_config(page_title="Studio Passport Maker", page_icon="📸", layout="centered")

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
    Studio Light Blue BG • Hair Safe • White Border • Signature Overlay
    </p>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("⚙️ Settings")
    st.info("Direct Offline Studio Engine (No API Key Required!)")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

def smart_crop_passport(img):
    h, w = img.shape[:2]
    target_ratio = 3.5 / 4.5
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x1 = max(0, (w - new_w) // 2)
        cropped = img[:, x1:x1 + new_w]
    else:
        new_h = int(w / target_ratio)
        y1 = max(0, int((h - new_h) * 0.12))
        cropped = img[y1:y1 + new_h, :]

    return cv2.resize(cropped, (413, 531), interpolation=cv2.INTER_AREA)

def apply_studio_blue_background(img):
    h, w = img.shape[:2]
    studio_blue = np.array([235, 185, 145], dtype=np.uint8)  # Light Sky Blue (BGR)

    # Center-weighted foreground protection (Hair and Body preservation)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    # Loose outer bounding rectangle
    rect = (12, 10, w - 24, h - 15)

    try:
        cv2.grabCut(img, mask, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
        
        # Protect head & hair region explicitly so top of head is never cut
        center_x, center_y = w // 2, int(h * 0.35)
        radius_x, radius_y = int(w * 0.36), int(h * 0.30)
        cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, cv2.GC_PR_FGD, -1)
        
        # Protect chest & shoulders
        cv2.rectangle(mask, (int(w * 0.10), int(h * 0.55)), (int(w * 0.90), h - 10), cv2.GC_PR_FGD, -1)

        # Generate binary mask
        fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Smooth edges for seamless blend
        fg_mask_blurred = cv2.GaussianBlur(fg_mask.astype(np.float32), (11, 11), 0)
        fg_3ch = np.repeat(fg_mask_blurred[:, :, np.newaxis], 3, axis=2)

        bg_canvas = np.full((h, w, 3), studio_blue, dtype=np.uint8)
        result = (img * fg_3ch + bg_canvas * (1.0 - fg_3ch)).astype(np.uint8)
        return result
    except Exception:
        return img

def apply_studio_lighting(img):
    try:
        smooth = cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=30)
        lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        l_boosted = cv2.convertScaleAbs(l_clahe, alpha=1.05, beta=6)
        enhanced_bgr = cv2.cvtColor(cv2.merge([l_boosted, a, b]), cv2.COLOR_LAB2BGR)
        return cv2.addWeighted(enhanced_bgr, 1.10, img, -0.10, 0)
    except Exception:
        return img

def add_white_inner_border(img):
    h, w = img.shape[:2]
    res = img.copy()
    cv2.rectangle(res, (10, 10), (w - 10, h - 10), (255, 255, 255), 4)
    return res

def draw_cursive_signature(pil_img, text="Pankaj Yadav"):
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size
    
    # Try system fonts, fallback to default
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 26)
    except Exception:
        font = ImageFont.load_default()

    # Draw natural signature overlay on chest area
    pos_x = int(w * 0.32)
    pos_y = int(h * 0.74)
    draw.text((pos_x, pos_y), text, fill=(20, 20, 20), font=font)
    return pil_img

photo_mode = st.radio("Photo kaise select karni hai?", ["📁 Gallery / File Upload", "📷 Live Camera"], horizontal=True)

final_photo_data = None
if photo_mode == "📁 Gallery / File Upload":
    uploaded_photo = st.file_uploader("1️⃣ Photo Upload Karein", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_photo:
        final_photo_data = uploaded_photo.read()
else:
    camera_photo = st.camera_input("1️⃣ Camera se photo click karein")
    if camera_photo:
        final_photo_data = camera_photo.read()

sig_text = st.text_input("Signature Name:", value="Pankaj Yadav")

if final_photo_data:
    if st.button("⚡ GENERATE STUDIO PASSPORT PHOTO NOW", type="primary", use_container_width=True):
        with st.spinner("Studio photo generate ho rahi hai..."):
            np_arr = np.asarray(bytearray(final_photo_data), dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # 1. Framing
            framed = smart_crop_passport(img)

            # 2. Safe Light Blue BG
            blue_bg = apply_studio_blue_background(framed)

            # 3. Studio Lighting & Glow
            glow = apply_studio_lighting(blue_bg)

            # 4. White Border
            bordered = add_white_inner_border(glow)

            # 5. Signature Overlay
            final_rgb = cv2.cvtColor(bordered, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(final_rgb)
            if sig_text.strip():
                pil_img = draw_cursive_signature(pil_img, sig_text.strip())

            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=95)
            byte_im = buf.getvalue()

            st.success("✅ Studio Passport Photo Taiyaar Ho Gayi!")
            st.image(byte_im, caption="Passport Preview (3.5cm x 4.5cm)", width=260)

            st.download_button(
                label="📥 DOWNLOAD PASSPORT PHOTO",
                data=byte_im,
                file_name="studio_passport.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
