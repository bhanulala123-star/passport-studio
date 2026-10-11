import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
import io

# --- SPEED OPTIMIZATION (LAZY LOADING) ---
# Heavy AI models ko UI load hote time block karne se rokne ke liye
@st.cache_resource
def get_ai_session():
    # rembg sirf tabhi import hoga jab zaroorat hogi, isse app link instantly open hoga
    from rembg import new_session
    return new_session("u2netp")

def detect_and_crop_face(image: Image.Image, target_ratio=3/4) -> Image.Image:
    """Detects face using OpenCV Haar Cascades exactly as requested."""
    import cv2  # Lazy import for fast initial app loading
    
    cv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    img_h, img_w = cv_img.shape[:2]
    
    face_found = False
    
    try:
        if hasattr(cv2, 'CascadeClassifier'):
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Detect face
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            
            if len(faces) > 0:
                # Get largest face
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                x, y, w, h = faces[0]
                
                # Target aspect ratio is 3:4 (w:h) - Subject takes 70-80% of frame
                target_h = int(h * 2.5) 
                target_w = int(target_h * target_ratio)
                
                center_x = x + w // 2
                center_y = y + h // 2
                
                crop_y1 = max(0, center_y - int(target_h * 0.45))
                crop_y2 = crop_y1 + target_h
                crop_x1 = max(0, center_x - target_w // 2)
                crop_x2 = crop_x1 + target_w
                
                # Handle boundaries
                if crop_y2 > img_h:
                    crop_y2 = img_h
                    crop_y1 = max(0, crop_y2 - target_h)
                if crop_x2 > img_w:
                    crop_x2 = img_w
                    crop_x1 = max(0, crop_x2 - target_w)
                    
                cropped_cv = cv_img[crop_y1:crop_y2, crop_x1:crop_x2]
                face_found = True
    except Exception:
        pass # Safe fallback for cloud environments

    # Fallback: Center crop to 3:4 ratio if OpenCV fails to find a face
    if not face_found:
        target_w = min(img_w, int(img_h * target_ratio))
        target_h = int(target_w / target_ratio)
        x_offset = (img_w - target_w) // 2
        y_offset = (img_h - target_h) // 2
        cropped_cv = cv_img[y_offset:y_offset+target_h, x_offset:x_offset+target_w]

    return Image.fromarray(cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB))

def process_signature(sig_image: Image.Image) -> Image.Image:
    """Strict background removal keeping ONLY dark ink using Otsu's thresholding."""
    import cv2  # Lazy import
    
    cv_sig = cv2.cvtColor(np.array(sig_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray_sig = cv2.cvtColor(cv_sig, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray_sig, (3, 3), 0)
    
    # Otsu's thresholding (inverse so ink is white, background is black)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Make everything but the ink 100% transparent (Alpha = 0)
    h, w = thresh.shape
    rgba_sig = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_sig[:, :, 0] = 0   
    rgba_sig[:, :, 1] = 0   
    rgba_sig[:, :, 2] = 51  # Dark blue ink
    rgba_sig[:, :, 3] = thresh # Alpha channel mask
    
    return Image.fromarray(rgba_sig, "RGBA")

def add_rounded_border(image: Image.Image, radius=20, border_width=4, color=(255, 255, 255)) -> Image.Image:
    """Overlays a thin elegant white border with rounded corners."""
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

def prevent_oom_crash(img: Image.Image, max_size=800) -> Image.Image:
    """Resizes large smartphone photos to prevent 1GB RAM limits on Streamlit."""
    img_copy = img.copy()
    img_copy.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img_copy

def main():
    st.title("Passport Photo & Signature Generator")
    st.write("Upload a portrait and a signature photo to generate a formatted passport size image.")
    
    col1, col2 = st.columns(2)
    with col1:
        main_photo = st.file_uploader("Upload Main Photo", type=["jpg", "jpeg", "png"])
    with col2:
        sig_photo = st.file_uploader("Upload Signature Photo", type=["jpg", "jpeg", "png"])

    if main_photo and sig_photo:
        if st.button("Generate Passport Photo"):
            with st.spinner("Processing studio-quality photo..."):
                try:
                    # Lazy loading rembg remove function
                    from rembg import remove
                    
                    # Session loaded instantly from cache
                    ai_session = get_ai_session()
                    
                    # 1. Load and safely resize photos
                    img = Image.open(main_photo).convert("RGB")
                    img = prevent_oom_crash(img, max_size=800)
                    
                    sig = Image.open(sig_photo).convert("RGB")
                    sig = prevent_oom_crash(sig, max_size=600)
                    
                    # 2. Smart Crop to Passport Size
                    cropped_img = detect_and_crop_face(img)
                    
                    # 3. Background Removal
                    subject_only = remove(cropped_img, session=ai_session).convert("RGBA")
                    
                    # Apply Pastel Light-Blue Background (#8ECAE6)
                    bg_color = (142, 202, 230, 255)
                    final_bg = Image.new("RGBA", subject_only.size, bg_color)
                    composite = Image.alpha_composite(final_bg, subject_only)
                    
                    # 4. Main Photo Enhancement (+5 to 10% brightness/contrast)
                    enhancer_b = ImageEnhance.Brightness(composite)
                    composite = enhancer_b.enhance(1.10)
                    enhancer_c = ImageEnhance.Contrast(composite)
                    composite = enhancer_c.enhance(1.05)
                    
                    # 5. Process Transparent Signature
                    transparent_sig = process_signature(sig)
                    
                    # 6. Place Signature at the bottom-center (55% width)
                    target_sig_w = int(composite.width * 0.55)
                    sig_ratio = target_sig_w / transparent_sig.width
                    target_sig_h = int(transparent_sig.height * sig_ratio)
                    transparent_sig = transparent_sig.resize((target_sig_w, target_sig_h), Image.Resampling.LANCZOS)
                    
                    paste_x = (composite.width - target_sig_w) // 2
                    paste_y = composite.height - target_sig_h - 20
                    composite.paste(transparent_sig, (paste_x, paste_y), mask=transparent_sig)
                    
                    # 7. Add Correct White Border Overlay
                    final_image = add_rounded_border(composite, radius=30, border_width=6)
                    
                    st.image(final_image, caption="Final Passport Photo", use_column_width=True)
                    
                    # Download button
                    buf = io.BytesIO()
                    final_image.convert("RGB").save(buf, format="JPEG", quality=95)
                    st.download_button("Download Processed Photo", data=buf.getvalue(), file_name="passport_photo.jpg", mime="image/jpeg")
                    
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")

if __name__ == "__main__":
    main()
