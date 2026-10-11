import io
import math
import os
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import streamlit as st
import cv2

# Streamlit/Cloud par OpenCV X11 errors ko rokne ke liye
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

def open_image(uploaded):
    """EXIF orientation fix karke image load karna"""
    im = Image.open(uploaded)
    return ImageOps.exif_transpose(im).convert("RGB")

def auto_straighten(img):
    """Haar Cascade ka use karke face/eyes detect karna aur photo ko seedha karna"""
    try:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        eye_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
        
        if not os.path.exists(cascade_path) or not os.path.exists(eye_path):
            return img, "Face model files unavailable. Original angle kept."
            
        faces = cv2.CascadeClassifier(cascade_path).detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70)
        )
        if len(faces) == 0:
            return img, "No face detected. Original angle kept."
            
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_gray = gray[y:y + int(h * 0.62), x:x + w]
        eyes = cv2.CascadeClassifier(eye_path).detectMultiScale(
            face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(12, 12)
        )
        
        centers = [(ex + ew / 2, y + ey + eh / 2) for ex, ey, ew, eh in eyes]
        centers = sorted(centers, key=lambda p: p[0])
        if len(centers) < 2:
            return img, "Both eyes not detected. Original angle kept."
            
        pairs = [(a, b) for i, a in enumerate(centers) for b in centers[i+1:]]
        left, right = max(pairs, key=lambda pair: pair[1][0] - pair[0][0])
        angle = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
        
        if abs(angle) < 0.7:
            return img, f"Face looks straight (estimated tilt {angle:.1f}°)."
        if abs(angle) > 25:
            return img, f"Tilt {angle:.1f}° too large to correct safely."
            
        fixed = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(245, 245, 245))
        return fixed, f"Straightened using eye-line estimate ({angle:.1f}°)."
    except Exception as e:
        return img, f"Auto-straighten skipped safely: {e}"

def remove_background(img):
    """rembg ka use karke background nikalna"""
    try:
        from rembg import remove
        out = remove(img)
        if isinstance(out, bytes):
            out = Image.open(io.BytesIO(out))
        return out.convert("RGBA"), None
    except Exception as e:
        return None, f"Background removal failed: {e}"

def enhance_photo(img, strength):
    """Conservative enhancement - skin texture preserve karta hai"""
    im = ImageOps.autocontrast(img.convert("RGB"), cutoff=0.4)
    im = ImageEnhance.Color(im).enhance(1.04 + 0.08 * strength)
    im = ImageEnhance.Contrast(im).enhance(1.03 + 0.07 * strength)
    im = ImageEnhance.Brightness(im).enhance(1.015)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.3 + strength * 0.6, percent=105 + int(strength * 45), threshold=3))
    return im

def make_studio_gradient(size, base_color):
    """Soft light-blue studio gradient background banana"""
    W, H = size
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W * 0.50, H * 0.42
    dist = np.sqrt(((xx - cx) / (W * 0.78))**2 + ((yy - cy) / (H * 0.78))**2)
    glow = np.clip(1.0 - dist, 0, 1)[:, :, None] * 20.0
    base = np.array(base_color, dtype=np.float32)[None, None, :]
    arr = np.clip(base + glow, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")

def fit_subject_on_background(rgba, bg_color, out_size, margin=0.06):
    """Subject ko center karke, shoulder properly fit karna"""
    W, H = out_size
    bg = make_studio_gradient((W, H), bg_color).convert("RGBA")
    subject = rgba.convert("RGBA")
    
    alpha = subject.getchannel("A")
    bbox = alpha.getbbox()
    if bbox:
        subject = subject.crop(bbox)
        
    max_w, max_h = int(W * (1 - 2 * margin)), int(H * (1 - margin))
    scale = min(max_w / max(1, subject.width), max_h / max(1, subject.height))
    
    new_w = max(1, int(subject.width * scale))
    new_h = max(1, int(subject.height * scale))
    subject = subject.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    a = subject.getchannel("A").filter(ImageFilter.GaussianBlur(0.35))
    subject.putalpha(a)
    
    x = (W - subject.width) // 2
    y = max(0, H - subject.height - int(H * 0.025))
    bg.alpha_composite(subject, (x, y))
    return bg.convert("RGB")

def process_signature(upload, threshold, thickness):
    """Signature ka background transparent banakar ink nikalna"""
    if upload is None:
        return None
    sig = Image.open(upload)
    sig = ImageOps.exif_transpose(sig).convert("RGBA")
    
    arr = np.array(sig).astype(np.int16)
    rgb = arr[:, :, :3]
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])
    
    ink = np.clip((threshold - gray) / max(1, threshold - 30) * 255, 0, 255).astype(np.uint8)
    alpha = Image.fromarray(ink, "L").filter(ImageFilter.GaussianBlur(0.35))
    
    sig_out = Image.new("RGBA", sig.size, (20, 20, 20, 0))
    sig_out.putalpha(alpha)
    bbox = sig_out.getchannel("A").getbbox()
    if bbox:
        sig_out = sig_out.crop(bbox)
        
    if thickness > 0:
        a = sig_out.getchannel("A").filter(ImageFilter.MaxFilter(3 if thickness == 1 else 5))
        sig_out.putalpha(a)
    return sig_out

def add_border_and_signature(photo, sig_img, border_size=15, include_sig=False, strip_height=140):
    """White border aur optional white signature strip lagana"""
    W, H = photo.size
    final_W = W + 2 * border_size
    final_H = H + 2 * border_size
    
    if include_sig and sig_img:
        final_H += strip_height
        
    canvas = Image.new("RGB", (final_W, final_H), "white")
    canvas.paste(photo, (border_size, border_size))
    
    if include_sig and sig_img:
        sig_w, sig_h = sig_img.size
        avail_h = strip_height - 30
        avail_w = final_W - 40
        scale = min(avail_w / max(1, sig_w), avail_h / max(1, sig_h))
        new_w, new_h = max(1, int(sig_w * scale)), max(1, int(sig_h * scale))
        sig_resized = sig_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        x = (final_W - new_w) // 2
        y = border_size + H + (strip_height - new_h) // 2
        canvas.paste(sig_resized, (x, y), sig_resized)
        
    return canvas

def main():
    st.set_page_config(page_title="Passport Studio HD", page_icon="📸", layout="centered")
    st.title("📸 Passport Studio HD")
    st.caption("Professional portrait maker with background cleanup, studio background, white border, and optional signature overlay.")
    st.info("For best results, upload a clear front-facing photo. Background removal uses AI.")
    
    uploaded = st.file_uploader("Upload Portrait Photo", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded:
        original = open_image(uploaded)
        # BUG FIXED HERE: Removed problematic width parameters completely
        st.image(original, caption="Original Upload")
        st.markdown("---")
        st.subheader("⚙️ Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            do_straighten = st.checkbox("Auto-straighten face", value=True)
            do_remove = st.checkbox("Remove Background (AI)", value=True)
            enhancement = st.slider("Enhancement Strength", 0.0, 1.0, 0.4)
        with col2:
            out_width = st.number_input("Photo Width (px)", value=600, step=50)
            out_height = st.number_input("Photo Height (px)", value=800, step=50)
            border_size = st.number_input("White Border Size (px)", value=15, step=5)
            
        st.markdown("---")
        st.subheader("✍️ Signature Options")
        sig_upload = st.file_uploader("Upload Signature (Optional, transparent or white bg)", type=["jpg", "jpeg", "png"])
        inc_sig = st.checkbox("Add Signature Strip at Bottom", value=bool(sig_upload))
        sig_thresh = st.slider("Signature Ink Threshold", 50, 250, 200)
        
        if st.button("Generate Passport Photo", type="primary"):
            with st.spinner("Processing... Please wait."):
                img = original.copy()
                if do_straighten:
                    img, msg = auto_straighten(img)
                    st.toast(msg)
                
                bg_color = (205, 222, 238)
                if do_remove:
                    rgba, err = remove_background(img)
                    if err:
                        st.error(err)
                        composed = img
                    else:
                        composed = fit_subject_on_background(rgba, bg_color, (out_width, out_height))
                else:
                    composed = img.resize((out_width, out_height), Image.Resampling.LANCZOS)
                
                enhanced = enhance_photo(composed, enhancement)
                
                sig_img = None
                if sig_upload and inc_sig:
                    sig_img = process_signature(sig_upload, sig_thresh, thickness=1)
                
                final = add_border_and_signature(enhanced, sig_img, border_size=int(border_size), include_sig=inc
