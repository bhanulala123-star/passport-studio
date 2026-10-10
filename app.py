import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rembg import remove
import io

st.set_page_config(page_title="Pro Passport Photo Maker", layout="centered")

def enhance_image_like_remini(image):
    # PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 1. Smooth skin but keep edges sharp (Glow & Smoothness)
    smooth = cv2.bilateralFilter(img_cv, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 2. Enhance Contrast and Brightness (CLAHE)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced_cv = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # 3. Slight Saturation boost for that fresh look
    hsv = cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 15) # Boost color slightly
    final_hsv = cv2.merge((h, s, v))
    final_cv = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    
    return Image.fromarray(final_cv)

def process_background_and_border(image, bg_color=(100, 175, 230)):
    # Remove Background
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    output_byte_arr = remove(img_byte_arr.getvalue())
    fg_img = Image.open(io.BytesIO(output_byte_arr)).convert("RGBA")
    
    # Create Blue Background
    bg = Image.new("RGBA", fg_img.size, bg_color + (255,))
    bg.paste(fg_img, (0, 0), fg_img)
    
    # Add White Border (Inset)
    final_img = bg.convert("RGB")
    draw = ImageDraw.Draw(final_img)
    border_width = int(final_img.width * 0.03)
    draw.rectangle(
        [border_width, border_width, final_img.width - border_width, final_img.height - border_width],
        outline="white", width=border_width
    )
    return final_img

def process_signature_image(sig_image):
    # Convert signature to grayscale and remove white background
    sig_cv = cv2.cvtColor(np.array(sig_image), cv2.COLOR_RGB2GRAY)
    
    # Thresholding to extract black ink only
    _, thresh = cv2.threshold(sig_cv, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Make signature bolder
    kernel = np.ones((2,2), np.uint8)
    bold_sig = cv2.dilate(thresh, kernel, iterations=1)
    
    # Convert back to RGBA for transparent overlay
    h, w = bold_sig.shape
    rgba_sig = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_sig[..., 3] = bold_sig # Alpha channel
    rgba_sig[..., 0:3] = 0 # Black color for signature
    
    return Image.fromarray(rgba_sig)

st.title("📸 Pro Passport & Sign Generator")
st.write("Apne cafe ke liye HD Passport photos banayein (Auto Enhance, Blue BG, White Border & Signature)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Photo Upload")
    uploaded_photo = st.file_uploader("Upload Person Photo", type=['jpg', 'jpeg', 'png'])

with col2:
    st.subheader("2. Signature Upload / Text")
    sig_option = st.radio("Signature Format:", ["Image Upload", "Type Text"])
    
    uploaded_sig = None
    sig_text = ""
    if sig_option == "Image Upload":
        uploaded_sig = st.file_uploader("Upload Signature", type=['jpg', 'jpeg', 'png'])
    else:
        sig_text = st.text_input("Enter Name for Signature (Cursive):", "Pankaj Yadav")

if uploaded_photo is not None:
    original_img = Image.open(uploaded_photo)
    st.image(original_img, caption="Original Photo", use_column_width=True)
    
    if st.button("Generate HD Passport Photo"):
        with st.spinner("Enhancing and Processing..."):
            # Step 1: Enhance Image (Glow & Smooth)
            enhanced_img = enhance_image_like_remini(original_img)
            
            # Step 2: Remove BG, Add Blue color & White Border
            passport_img = process_background_and_border(enhanced_img)
            
            # Step 3: Add Signature
            if sig_option == "Image Upload" and uploaded_sig is not None:
                sig_img = Image.open(uploaded_sig)
                processed_sig = process_signature_image(sig_img)
                
                # Resize signature to fit bottom
                sig_width = int(passport_img.width * 0.7)
                sig_ratio = sig_width / float(processed_sig.width)
                sig_height = int(float(processed_sig.height) * float(sig_ratio))
                processed_sig = processed_sig.resize((sig_width, sig_height), Image.LANCZOS)
                
                # Paste at bottom center
                x = (passport_img.width - processed_sig.width) // 2
                y = passport_img.height - processed_sig.height - 20
                passport_img.paste(processed_sig, (x, y), processed_sig)
                
            elif sig_option == "Type Text" and sig_text:
                draw = ImageDraw.Draw(passport_img)
                # Note: Aapko ek cursive TTF font download karke yahan path dena hoga
                try:
                    font = ImageFont.truetype("BrushScriptMT.ttf", int(passport_img.width * 0.1))
                except:
                    font = ImageFont.load_default() # Fallback
                
                # Calculate text size (approximate center)
                x = passport_img.width // 4
                y = passport_img.height - int(passport_img.height * 0.15)
                draw.text((x, y), sig_text, fill="black", font=font)

            st.success("Photo Ready!")
            st.image(passport_img, caption="Final HD Passport Photo", use_column_width=True)
            
            # Download Button
            buf = io.BytesIO()
            passport_img.save(buf, format="PNG", quality=100)
            byte_im = buf.getvalue()
            st.download_button(label="Download HD Photo", data=byte_im, file_name="passport_ready.png", mime="image/png")
