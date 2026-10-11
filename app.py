import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
from rembg import remove, new_session
import io

def detect_and_crop_face(image: Image.Image, target_ratio=3/4) -> Image.Image:
    """Detects face using OpenCV Haar Cascades exactly as requested, with a safe fallback."""
    cv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    img_h, img_w = cv_img.shape[:2]
    
    face_found = False
    
    # Safe strict implementation of your OpenCV Haar Cascade logic
    try:
        if hasattr(cv2, 'CascadeClassifier'):
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            
            if len(faces) > 0:
                # Get largest face
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                x, y, w, h = faces[0]
                
                # Target aspect ratio is 3:4 (w:h) - Face takes ~70-80%
                target_h = int(h * 2.5) 
                target_w = int(target_h * target_ratio)
                
                center_x = x + w // 2
                center_y = y + h // 2
                
                crop_y1 = max(0, center_y - int(target_h * 0.45))
                crop_y2 = crop_y1 + target_h
                crop_x1 = max(0, center_x - target_w // 2)
                crop_x2 = crop_x1 + target_w
                
                if crop_y2 > img_h:
                    crop_y2 = img_h
                    crop_y1 = max(0, crop_y2 - target_h)
                if crop_x2 > img_w:
                    crop_x2 = img_w
                    crop_x1 = max(0, crop_x2 - target_w)
                    
                cropped_cv = cv_img[crop_y1:crop_y2, crop_x1:crop_x2]
                face_found = True
    except Exception:
        pass # Fallback par jayega agar cloud mein file missing hui

    # Fallback Option (Agar face detect nahi hua)
    if not face_found:
        target_w = min(img_w, int(img_h * target_ratio))
        target_h = int(target_w / target_ratio)
        x_offset = (img_w - target_w) // 2
        y_offset = (img_h - target_h) // 2
        cropped_cv = cv_img[y_offset:y_offset+target_h, x_offset:x_offset+target_w]

    return Image.fromarray(cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB))

def process_signature(sig_image: Image.Image) -> Image.Image:
    """Strict Signature Background Removal using Otsu's thresholding."""
    cv_sig = cv2.cvtColor(np.array(sig_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray_sig = cv2.cvtColor(cv_sig, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray_sig, (3, 3), 0)
    
    # Otsu's thresholding (inverse)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Transparent background logic as requested
    h, w = thresh.shape
    rgba_sig = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_sig[:, :, 0] = 0   
    rgba_sig[:, :, 1] = 0   
    rgba_sig[:, :, 2] = 51  # Dark blue ink
    rgba_sig[:, :, 3] = thresh # Alpha channel is 0 for background
    
    return Image.fromarray(rgba_sig, "RGBA")

def add_rounded_border(image: Image.Image, radius=20, border_width=4, color=(255, 255, 255)) -> Image.Image:
    """Overlays a thin white border with rounded corners."""
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), image.size], radius=radius, fill=255)
    
    rounded_img = Image.new('RGBA', image.size, (0,0,0,0))
    rounded_img.paste(image, (0,0), mask=mask)
    
    border_overlay = Image.new('RGBA', image.size, (0,0,0,0))
    b_draw = ImageDraw.Draw(border_overlay)
    b_draw.rounded_rectangle([(border_width//2, border_width//2), 
                              (image.size[0]-border_width//2, image.size[1]-border_width//2)], 
                             radius=radius, outline=color, width=border_width)
    
    return Image.alpha_composite(rounded_img, border_overlay)

def prevent_oom_crash(img: Image.Image, max_size=900) -> Image.Image:
    """Slightly resizes images before AI processing to fit inside 1GB RAM."""
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        return img.resize((int(img.width * ratio), int(img.height * ratio)), Image.Resampling.LANCZOS)
    return img

def main():
    st.title("Passport Photo & Signature Generator")
    
    col1, col2 = st.columns(2)
    with col1:
        main_photo = st.file_uploader("Upload Main Photo", type=["jpg", "jpeg", "png"])
    with col2:
        sig_photo = st.file_uploader("Upload Signature Photo", type=["jpg", "jpeg", "png"])

    if main_photo and sig_photo:
        if st.button("Generate Passport Photo"):
            with st.spinner("Processing images (Please wait a few seconds)..."):
                try:
                    # Initialize lightweight u2net session so it DOES NOT download the 1GB bria-rmbg model
                    ai_session = new_session("u2net")
                    
                    # 1. Load and resize
                    img = Image.open(main_photo).convert("RGB")
                    img = prevent_oom_crash(img)
                    
                    sig = Image.open(sig_photo).convert("RGB")
                    sig = prevent_oom_crash(sig, max_size=600)
                    
                    # 2. Smart Crop to Passport Size (using your Haar Cascade logic)
                    cropped_img = detect_and_crop_face(img)
                    
                    # 3. Background Removal (with strict memory safety)
                    subject_only = remove(cropped_img, session=ai_session).convert("RGBA")
                    
                    # Apply Pastel Blue Background (#8ECAE6)
                    bg_color = (142, 202, 230, 255)
                    final_bg = Image.new("RGBA", subject_only.size, bg_color)
                    composite = Image.alpha_composite(final_bg, subject_only)
                    
                    # 4. Enhance Main Photo (PIL ImageEnhance)
                    enhancer_b = ImageEnhance.Brightness(composite)
                    composite = enhancer_b.enhance(1.10)
                    enhancer_c = ImageEnhance.Contrast(composite)
                    composite = enhancer_c.enhance(1.05)
                    
                    # 5. Strict Signature Processing (Thresholding & Transparency)
                    transparent_sig = process_signature(sig)
                    
                    # 6. Place Signature
                    target_sig_w = int(composite.width * 0.55)
                    sig_ratio = target_sig_w / transparent_sig.width
                    target_sig_h = int(transparent_sig.height * sig_ratio)
                    transparent_sig = transparent_sig.resize((target_sig_w, target_sig_h), Image.Resampling.LANCZOS)
                    
                    paste_x = (composite.width - target_sig_w) // 2
                    paste_y = composite.height - target_sig_h - 20
                    composite.paste(transparent_sig, (paste_x, paste_y), mask=transparent_sig)
                    
                    # 7. Add Correct Border Overlay
                    final_image = add_rounded_border(composite, radius=30, border_width=6)
                    
                    # Handle updated Streamlit parameter
                    st.image(final_image, caption="Final Passport Photo", use_column_width=True)
                    
                    # Download button
                    buf = io.BytesIO()
                    final_image.convert("RGB").save(buf, format="JPEG", quality=95)
                    st.download_button("Download Processed Photo", data=buf.getvalue(), file_name="passport_photo.jpg", mime="image/jpeg")
                    
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")

if __name__ == "__main__":
    main()
