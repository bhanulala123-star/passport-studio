import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rembg import remove
import io
import os

st.set_page_config(page_title="Pro Passport Photo Maker", layout="centered")

# --- Step 1: Face Detection & Center Cropping ---
def auto_center_crop(image):
    # PIL image ko OpenCV mein badalna
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Pre-trained classifier file ko load karna. (Streamlit mein yahi best tarika hai)
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    # Face detect karna
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return image # Agar face nahi mila toh original image hi dega
    
    # Sabse bada face uthana (agar 2 log hain toh)
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    x, y, w, h = faces[0]
    
    # Face ko center mein rakh ke Passport Size (3:4 ratio) ke hisab se crop karna
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Passport standard ratio 3:4 (width:height) set karna
    crop_width = int(w * 2.5) 
    crop_height = int(crop_width * 1.33)
    
    x1 = max(0, center_x - crop_width // 2)
    y1 = max(0, center_y - int(h * 0.8)) # Thoda upar se katna
    x2 = min(img_cv.shape[1], x1 + crop_width)
    y2 = min(img_cv.shape[0], y1 + crop_height)
    
    cropped_cv = img_cv[y1:y2, x1:x2]
    return Image.fromarray(cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB))


# --- Step 2: Remini-like Enhancement (Glow & Texture) ---
def enhance_image_like_remini(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Skin smooth karna lekin edges sharp rakhna
    smooth = cv2.bilateralFilter(img_cv, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Contrast aur Brightness badhana
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced_cv = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Glow (Saturation) badhana thoda sa
    hsv = cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 15) 
    final_hsv = cv2.merge((h, s, v))
    final_cv = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    
    return Image.fromarray(final_cv)


# --- Step 3: Background Remove, Blue BG aur White Border ---
def process_background_and_border(image, bg_color=(100, 175, 230)):
    # Rembg se Background hatana
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    output_byte_arr = remove(img_byte_arr.getvalue())
    fg_img = Image.open(io.BytesIO(output_byte_arr)).convert("RGBA")
    
    # Naya Blue background (Exact aapke photo jaisa color)
    bg = Image.new("RGBA", fg_img.size, bg_color + (255,))
    bg.paste(fg_img, (0, 0), fg_img)
    
    # Photo ko standard size mein resize karna
    bg = bg.resize((600, 800), Image.LANCZOS)
    
    # Border lagana (Aapki mangi hui white border)
    final_img = bg.convert("RGB")
    draw = ImageDraw.Draw(final_img)
    border_width = 15 # Fixed border size
    draw.rectangle(
        [border_width, border_width, final_img.width - border_width, final_img.height - border_width],
        outline="white", width=border_width
    )
    return final_img


# --- Step 4: Signature ko transparent karke photo pe dalna ---
def process_signature_image(sig_image):
    sig_cv = cv2.cvtColor(np.array(sig_image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(sig_cv, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2,2), np.uint8)
    bold_sig = cv2.dilate(thresh, kernel, iterations=1)
    
    h, w = bold_sig.shape
    rgba_sig = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_sig[..., 3] = bold_sig
    rgba_sig[..., 0:3] = 0 # Black ink only
    
    return Image.fromarray(rgba_sig)


# --- UI Design ---
st.title("📸 Pro Passport & Sign Generator")
st.write("Apne cafe ke liye HD Passport photos banayein. (Auto Center Crop, Blue BG, White Border & Sign)")

col1, col2 = st.columns(2)
with col1:
    uploaded_photo = st.file_uploader("1. Photo Upload (Kaisi bhi ho)", type=['jpg', 'jpeg', 'png'])

with col2:
    sig_option = st.radio("2. Signature Format:", ["Image Upload", "Type Text"])
    uploaded_sig = None
    sig_text = ""
    if sig_option == "Image Upload":
        uploaded_sig = st.file_uploader("Upload Signature", type=['jpg', 'jpeg', 'png'])
    else:
        sig_text = st.text_input("Enter Name (Cursive Text):", "Pankaj Yadav")

if uploaded_photo is not None:
    original_img = Image.open(uploaded_photo)
    st.image(original_img, caption="Original Photo", use_column_width=True)
    
    if st.button("Generate HD Passport Photo"):
        with st.spinner("Processing: Auto-cropping, Enhancing & Adding Blue BG..."):
            
            # 1. Chehra dhund ke Center me Crop
            cropped_img = auto_center_crop(original_img)
            
            # 2. Glow aur Smoothness
            enhanced_img = enhance_image_like_remini(cropped_img)
            
            # 3. BG Remove aur Blue + Border
            passport_img = process_background_and_border(enhanced_img)
            
            # 4. Signature Lagana
            if sig_option == "Image Upload" and uploaded_sig is not None:
                sig_img = Image.open(uploaded_sig)
                processed_sig = process_signature_image(sig_img)
                
                sig_width = int(passport_img.width * 0.7)
                sig_ratio = sig_width / float(processed_sig.width)
                sig_height = int(float(processed_sig.height) * float(sig_ratio))
                processed_sig = processed_sig.resize((sig_width, sig_height), Image.LANCZOS)
                
                x = (passport_img.width - processed_sig.width) // 2
                y = passport_img.height - processed_sig.height - 30
                passport_img.paste(processed_sig, (x, y), processed_sig)
                
            elif sig_option == "Type Text" and sig_text:
                draw = ImageDraw.Draw(passport_img)
                try:
                    # Agar cloud pe ye font na ho, toh default chuna jayega
                    font = ImageFont.truetype("BrushScriptMT.ttf", 60)
                except:
                    font = ImageFont.load_default()
                
                x = passport_img.width // 4
                y = passport_img.height - 100
                draw.text((x, y), sig_text, fill="black", font=font)

            st.success("Ekdum Kadak Photo Ready!")
            st.image(passport_img, caption="Final HD Passport", use_column_width=True)
            
            buf = io.BytesIO()
            passport_img.save(buf, format="PNG", quality=100)
            byte_im = buf.getvalue()
            st.download_button(label="Download HD Photo", data=byte_im, file_name="passport_ready.png", mime="image/png")
