import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from rembg import remove
import io

# --- Helper Functions ---

def enhance_portrait(img: Image.Image) -> Image.Image:
    """Enhance the portrait for a clean studio look."""
    # 1. Boost Sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.3)
    # 2. Boost Brightness slightly
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.1)
    # 3. Boost Contrast slightly
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.15)
    return img

def remove_signature_background(sig_img: Image.Image) -> Image.Image:
    """Removes the white background from a signature, leaving only the ink."""
    # Convert PIL Image to OpenCV format (RGBA)
    sig_img = sig_img.convert("RGBA")
    np_img = np.array(sig_img)
    
    # Convert to grayscale to identify white background vs dark ink
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGBA2GRAY)
    
    # Apply a threshold: pixels darker than 220 become 255 (white/solid mask), lighter become 0 (transparent)
    _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    
    # Optional: Apply slight blur to the mask to anti-alias the edges of the ink
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # Apply the calculated mask to the Alpha (transparency) channel
    np_img[:, :, 3] = mask
    
    # Convert back to PIL Image
    return Image.fromarray(np_img)

# --- Streamlit UI & Main Logic ---

st.set_page_config(page_title="Self-Attested Photo Generator", layout="centered")

st.title("📸 Studio Passport & Signature Attestation")
st.write("Upload your portrait and signature to generate a self-attested studio photo with a blue background.")

col1, col2 = st.columns(2)
with col1:
    main_upload = st.file_uploader("1. Upload Main Portrait Photo", type=["jpg", "jpeg", "png"])
with col2:
    sig_upload = st.file_uploader("2. Upload Signature Image", type=["jpg", "jpeg", "png"])

if main_upload and sig_upload:
    st.markdown("---")
    st.write("### Processing...")
    
    with st.spinner("Removing background and enhancing portrait..."):
        # 1. Load and enhance the main photo
        main_img = Image.open(main_upload).convert("RGB")
        main_img = enhance_portrait(main_img)
        
        # 2. Remove background using rembg (returns RGBA image)
        main_no_bg = remove(main_img)
        
        # 3. Create a solid light blue background (standard studio passport blue)
        # Hex: #ADD8E6 -> RGB: (173, 216, 230)
        blue_bg = Image.new("RGB", main_no_bg.size, (173, 216, 230))
        
        # Paste the subject onto the blue background using the subject's alpha channel as the mask
        blue_bg.paste(main_no_bg, (0, 0), main_no_bg)
        final_portrait = blue_bg  # Now a solid RGB image
        
    with st.spinner("Processing signature and merging..."):
        # 4. Load and process signature
        sig_img_raw = Image.open(sig_upload)
        sig_transparent = remove_signature_background(sig_img_raw)
        
        # 5. Overlaying the signature
        # Resize signature to fit 80% of the portrait's width
        target_width = int(final_portrait.width * 0.8)
        aspect_ratio = sig_transparent.height / sig_transparent.width
        target_height = int(target_width * aspect_ratio)
        
        # Use LANCZOS for high-quality downsampling
        sig_resized = sig_transparent.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # Calculate coordinates to place signature at the bottom center (overlapping clothing)
        x_offset = int((final_portrait.width - target_width) / 2)
        # 10 pixels padding from the absolute bottom edge
        y_offset = final_portrait.height - target_height - 10 
        
        # Paste the transparent signature onto the portrait
        final_portrait.paste(sig_resized, (x_offset, y_offset), sig_resized)
        
    st.success("Success! Image generated.")
    
    # --- Final Output ---
    st.image(original, caption="Original Upload", use_container_width=True)

    # Prepare image for download
    buf = io.BytesIO()
    final_portrait.save(buf, format="JPEG", quality=95)
    byte_im = buf.getvalue()
    
    st.download_button(
        label="⬇️ Download High-Quality Photo",
        data=byte_im,
        file_name="self_attested_passport.jpg",
        mime="image/jpeg",
        use_container_width=True
    )
