import io
import math
import os
from PIL import Image, ImageOps, ImageEnhance, ImageDraw
import numpy as np
import streamlit as st
import cv2

# Streamlit/Cloud par OpenCV X11 errors ko rokne ke liye
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

def open_image(uploaded):
    im = Image.open(uploaded)
    return ImageOps.exif_transpose(im).convert("RGB")

def auto_straighten(img):
    try:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        eye_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
        
        if not os.path.exists(cascade_path) or not os.path.exists(eye_path):
            return img, "Face model files unavailable."
            
        faces = cv2.CascadeClassifier(cascade_path).detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70)
        )
        if len(faces) == 0:
            return img, "No face detected."
            
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_gray = gray[y:y + int(h * 0.62), x:x + w]
        eyes = cv2.CascadeClassifier(eye_path).detectMultiScale(
            face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(12, 12)
        )
        
        centers = [(ex + ew / 2, y + ey + eh / 2) for ex, ey, ew, eh in eyes]
        centers = sorted(centers, key=lambda p: p[0])
        if len(centers) < 2:
            return img, "Both eyes not detected."
            
        pairs = [(a, b) for i, a in enumerate(centers) for b in centers[i+1:]]
        left, right = max(pairs, key=lambda pair: pair[1][0] - pair[0][0])
        angle = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
        
        if abs(angle) < 0.7 or abs(angle) > 25:
            return img, "No safe tilt detected."
            
        fixed = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(245, 245, 245))
        return fixed, f"Straightened ({angle:.1f}°)."
    except Exception as e:
        return img, "Straighten skipped."

def remove_background(img):
    try:
        from rembg import remove
        out = remove(img)
        if isinstance(out, bytes):
            out = Image.open(io.BytesIO(out))
        return out.convert("RGBA"), None
    except Exception as e:
        return None, f"Background removal failed: {e}"

def enhance_photo(img, strength):
    """Natural look ke liye heavy filters hata diye gaye hain."""
    if strength == 0.0:
        return img
    # Sirf halki si brightness aur color, koi plastic look nahi
    im = ImageEnhance.Color(img).enhance(1.0 + (0.2 * strength))
    im = ImageEnhance.Brightness(im).enhance(1.0 + (0.1 * strength))
    return im

def make_solid_background(size, base_color):
    """Reference image jaisa clean solid blue background"""
    W, H = int(size[0]), int(size[1])
    bg = Image.new("RGB", (W, H), base_color)
    return bg

def fit_subject_on_background(rgba, bg_color, out_size, margin=0.04):
    W, H = int(out_size[0]), int(out_size[1])
    bg = make_solid_background((W, H), bg_color).convert("RGBA")
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
    
    x = int((W - subject.width) // 2)
    y = int(max(0, H - subject.height)) # Bottom se attach
    bg.alpha_composite(subject, (x, y))
    return bg.convert("RGB")

def process_signature(upload, threshold):
    if upload is None:
        return None
    sig = Image.open(upload)
    sig = ImageOps.exif_transpose(sig).convert("RGBA")
    
    arr = np.array(sig).astype(np.int16)
    rgb = arr[:, :, :3]
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])
    
    ink = np.clip((threshold - gray) / max(1, threshold - 30) * 255, 0, 255).astype(np.uint8)
    sig_out = Image.new("RGBA", sig.size, (15, 15, 15, 0)) # Dark signature ink
    sig_out.putalpha(Image.fromarray(ink, "L"))
    
    bbox = sig_out.getchannel("A").getbbox()
    if bbox:
        sig_out = sig_out.crop(bbox)
    return sig_out

def add_border_and_signature(photo, sig_img, border_size=15, include_sig=False):
    """Reference jaisa fine border aur bottom signature overlap"""
    border_size = int(border_size)
    W, H = photo.size
    
    # White base canvas
    final_W = int(W + 2 * border_size)
    final_H = int(H + 2 * border_size)
    
    # Agar signature chahiye toh niche thodi aur jagah
    if include_sig and sig_img:
        final_H += 80 
        
    canvas = Image.new("RGB", (final_W, final_H), "white")
    canvas.paste(photo, (border_size, border_size))
    
    # Draw inner rounded line like reference images
    draw = ImageDraw.Draw(canvas)
    inner_margin = border_size + 10
    draw.rounded_rectangle(
        [inner_margin, inner_margin, final_W - inner_margin, final_H - (100 if include_sig else inner_margin)],
        radius=20, outline="white", width=3
    )
    
    if include_sig and sig_img:
        sig_w, sig_h = sig_img.size
        avail_h = 100
        avail_w = final_W - 40
        scale = min(avail_w / max(1, sig_w), avail_h / max(1, sig_h))
        new_w, new_h = max(1, int(sig_w * scale)), max(1, int(sig_h * scale))
        sig_resized = sig_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Bottom center placement
        x = int((final_W - new_w) // 2)
        y = int(final_H - new_h - 15)
        
        # Paste with transparency
        canvas.paste(sig_resized, (x, y), sig_resized)
        
    return canvas

def main():
    st.set_page_config(page_title="Passport Studio HD", page_icon="📸", layout="centered")
    st.title("📸 Passport Studio HD")
    st.caption("Professional portrait maker with background cleanup, studio background, white border, and optional signature overlay.")
    
    uploaded = st.file_uploader("Upload Portrait Photo", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded:
        original = open_image(uploaded)
        st.image(original, caption="Original Upload")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            do_straighten = st.checkbox("Auto-straighten face", value=True)
            do_remove = st.checkbox("Remove Background (AI)", value=True)
            enhancement = st.slider("Enhancement (Natural)", 0.0, 1.0, 0.1) # Default low
        with col2:
            out_width = int(st.number_input("Photo Width (px)", value=600, step=50))
            out_height = int(st.number_input("Photo Height (px)", value=800, step=50))
            border_size = int(st.number_input("White Border Size (px)", value=15, step=5))
            
        st.markdown("---")
        sig_upload = st.file_uploader("Upload Signature (Optional)", type=["jpg", "jpeg", "png"])
        inc_sig = st.checkbox("Add Signature", value=bool(sig_upload))
        sig_thresh = st.slider("Signature Ink Threshold", 50, 250, 200)
        
        if st.button("Generate Passport Photo", type="primary"):
            with st.spinner("Processing... Please wait."):
                img = original.copy()
                if do_straighten:
                    img, msg = auto_straighten(img)
                
                # Reference image jaisa Professional Solid Blue color
                bg_color = (130, 185, 230) 
                
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
                    sig_img = process_signature(sig_upload, sig_thresh)
                
                final = add_border_and_signature(enhanced, sig_img, border_size=border_size, include_sig=inc_sig)
                
                st.success("Your photo is ready!")
                st.image(final, caption="Final Passport Photo")
                
                buf = io.BytesIO()
                final.save(buf, format="PNG")
                st.download_button(
                    label="Download HD PNG", 
                    data=buf.getvalue(), 
                    file_name="passport_ready.png", 
                    mime="image/png"
                )

if __name__ == '__main__':
    main()
