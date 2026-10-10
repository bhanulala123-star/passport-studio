import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import io
from rembg import remove

# UI sabse pehle load karne ke liye page config top par
st.set_page_config(page_title="Pro Passport Maker", layout="centered")

def auto_center_crop(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return image
    
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    x, y, w, h = faces[0]
    
    center_x = x + w // 2
    center_y = y + h // 2
    
    crop_width = int(w * 2.5) 
    crop_height = int(crop_width * 1.33)
    
    x1 = max(0, center_x - crop_width // 2)
    y1 = max(0, center_y - int(h * 0.8))
    x2 = min(img_cv.shape[1], x1 + crop_width)
    y2 = min(img_cv.shape[0], y1 + crop_height)
    
    cropped_cv = img_cv[y1:y2, x1:x2]
    return Image.fromarray(cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB))

def enhance_image(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    smooth = cv2.bilateralFilter(img_cv, d=9, sigmaColor=75, sigmaSpace=75)
    
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced_cv = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    hsv = cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 15) 
    final_hsv = cv2.merge((h, s, v))
    final_cv = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    
    return Image.fromarray(final_cv)

def process_background_and_border(image, bg_color=(100, 175, 230)):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    output_byte_arr = remove(img_byte_arr.getvalue())
    fg_img = Image.open(io.BytesIO(output_byte_arr)).convert("RGBA")
    
    bg = Image.new("RGBA", fg_img.size, bg_color + (255,))
    bg.paste(fg_img, (0, 0), fg_img)
    bg = bg.resize((600, 800), Image.LANCZOS)
    
    final_img = bg.convert("RGB")
    draw = ImageDraw.Draw(final_img)
    border_width = 15
    draw.rectangle(
        [border_width, border_width, final_img.width - border_width, final_img.height - border_width],
        outline="white", width=border_width
    )
    return final_img

def process_signature(sig_image):
    sig_cv = cv2.cvtColor(np.array(sig_image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(sig_cv, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2,2), np.uint8)
    bold_sig = cv2.dilate(thresh, kernel, iterations=1)
    
    h, w = bold_sig.shape
    rgba_sig = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_sig[..., 3] = bold_sig
    rgba_sig[..., 0:3] = 0 
    
    return Image.fromarray(rgba_sig)

st.title("📸 Twinkle Online Centre - Pro Passport Maker")
st.write("Upload photo and generate HD Passport Size Image with Auto-Crop, Blue BG, and Signature.")

col1, col2 = st.columns(2)
with col1:
    uploaded_photo = st.file_uploader("1. Upload Customer Photo", type=['jpg', 'jpeg', 'png'])

with col2:
    sig_option = st.radio("2. Signature Option:", ["Upload Image", "Type Name"])
    uploaded_sig = None
    sig_text = ""
    if sig_option == "Upload Image":
        uploaded_sig = st.file_uploader("Upload Signature", type=['jpg', 'jpeg', 'png'])
    else:
        sig_text = st.text_input("Enter Name (Cursive):", "Twinkle Yadav")

if uploaded_photo is not None:
    original_img = Image.open(uploaded_photo)
    st.image(original_img, caption="Original", use_column_width=True)
    
    if st.button("Generate HD Passport Photo"):
        with st.spinner("Processing Photo (Wait a few seconds)..."):
            cropped = auto_center_crop(original_img)
            enhanced = enhance_image(cropped)
            final_passport = process_background_and_border(enhanced)
            
            if sig_option == "Upload Image" and uploaded_sig is not None:
                sig_img = Image.open(uploaded_sig)
                proc_sig = process_signature(sig_img)
                
                sig_w = int(final_passport.width * 0.7)
                sig_ratio = sig_w / float(proc_sig.width)
                sig_h = int(float(proc_sig.height) * float(sig_ratio))
                proc_sig = proc_sig.resize((sig_w, sig_h), Image.LANCZOS)
                
                x = (final_passport.width - proc_sig.width) // 2
                y = final_passport.height - proc_sig.height - 30
                # Yahan par missing bracket thik kar diya gaya hai
                final_passport.paste(proc_sig, (x, y), proc_sig)
                
            elif sig_option == "Type Name" and sig_text:
                draw = ImageDraw.Draw(final_passport)
                try:
                    font = ImageFont.truetype("BrushScriptMT.ttf", 60)
                except:
                    font = ImageFont.load_default()
                x = final_passport.width // 4
                y = final_passport.height - 100
                draw.text((x, y), sig_text, fill="black", font=font)

            st.success("Photo is Ready!")
            st.image(final_passport, caption="Final Output", use_column_width=True)
            
            buf = io.BytesIO()
            final_passport.save(buf, format="PNG", quality=100)
            st.download_button(label="Download HD Photo", data=buf.getvalue(), file_name="passport_ready.png", mime="image/png")
