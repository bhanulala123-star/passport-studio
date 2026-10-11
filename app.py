import io
import math
import os
import textwrap
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import numpy as np
import streamlit as st

st.set_page_config(page_title="Passport Studio HD", page_icon="📸", layout="centered")

# Optional computer-vision/background-removal packages are imported only when needed.
def open_image(uploaded):
    im = Image.open(uploaded)
    return ImageOps.exif_transpose(im).convert("RGB")

def load_cv2():
    try:
        import cv2
        return cv2
    except Exception:
        return None

def auto_straighten(img):
    """Estimate head tilt from eye centers when OpenCV's bundled Haar models are available."""
    cv2 = load_cv2()
    if cv2 is None:
        return img, "Auto-straighten unavailable (OpenCV failed to load); kept original angle."
    try:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        eye_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
        if not os.path.exists(cascade_path) or not os.path.exists(eye_path):
            return img, "Face model files unavailable; kept original angle."
        faces = cv2.CascadeClassifier(cascade_path).detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70)
        )
        if len(faces) == 0:
            return img, "No face detected; kept original angle."
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_gray = gray[y:y + int(h * .62), x:x + w]
        eyes = cv2.CascadeClassifier(eye_path).detectMultiScale(
            face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(12, 12)
        )
        # Use two eye detections in the upper half, choosing the most separated pair.
        centers = [(x + ex + ew / 2, y + ey + eh / 2) for ex, ey, ew, eh in eyes]
        centers = sorted(centers, key=lambda p: p[0])
        if len(centers) < 2:
            return img, "Face found, but both eyes weren't detected; kept original angle."
        pairs = [(a, b) for i, a in enumerate(centers) for b in centers[i+1:]]
        left, right = max(pairs, key=lambda pair: pair[1][0] - pair[0][0])
        angle = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
        if abs(angle) < 0.7:
            return img, f"Face looks straight (estimated tilt {angle:.1f}°)."
        if abs(angle) > 25:
            return img, f"Estimated tilt {angle:.1f}° is too large to correct safely; kept original."
        fixed = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(245, 245, 245))
        return fixed, f"Straightened using eye-line estimate ({angle:.1f}°). Check preview before downloading."
    except Exception as e:
        return img, f"Auto-straighten skipped safely: {type(e).__name__}."

def remove_background(img):
    try:
        from rembg import remove
        out = remove(img)
        if isinstance(out, bytes):
            out = Image.open(io.BytesIO(out))
        return out.convert("RGBA"), None
    except Exception as e:
        return None, f"Background removal failed: {type(e).__name__}: {e}"

def enhance_photo(img, strength):
    # Conservative enhancement: preserves identity/details; not generative face replacement.
    im = ImageOps.autocontrast(img.convert("RGB"), cutoff=0.4)
    im = ImageEnhance.Color(im).enhance(1.04 + 0.08 * strength)
    im = ImageEnhance.Contrast(im).enhance(1.03 + 0.07 * strength)
    im = ImageEnhance.Brightness(im).enhance(1.015)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.3 + strength * .6, percent=105 + int(strength * 45), threshold=3))
    return im

def make_studio_gradient(size, base_color):
    """Soft light-center studio background, close to the supplied reference."""
    W, H = size
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W * 0.50, H * 0.42
    dist = np.sqrt(((xx - cx) / (W * 0.78))**2 + ((yy - cy) / (H * 0.78))**2)
    glow = np.clip(1.0 - dist, 0, 1)[:, :, None] * 20.0
    base = np.array(base_color, dtype=np.float32)[None, None, :]
    arr = np.clip(base + glow, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")

def fit_subject_on_background(rgba, bg_color, out_size, margin=0.06):
    W, H = out_size
    bg = make_studio_gradient((W, H), bg_color).convert("RGBA")
    subject = rgba.convert("RGBA")
    # Crop only transparent padding; do not crop face/hair/clothing based on guessed coordinates.
    alpha = subject.getchannel("A")
    bbox = alpha.getbbox()
    if bbox:
        subject = subject.crop(bbox)
    max_w, max_h = int(W * (1 - 2 * margin)), int(H * (1 - 2 * margin))
    scale = min(max_w / max(1, subject.width), max_h / max(1, subject.height))
    subject = subject.resize((max(1, int(subject.width * scale)), max(1, int(subject.height * scale))), Image.Resampling.LANCZOS)
    # Mild alpha-edge smoothing reduces jagged cut-outs without blurring the subject.
    a = subject.getchannel("A").filter(ImageFilter.GaussianBlur(0.35))
    subject.putalpha(a)
    x = (W - subject.width) // 2
    y = max(0, H - subject.height - int(H * 0.025))
    bg.alpha_composite(subject, (x, y))
    return bg.convert("RGB")

def find_font(size, bold=False, italic=False):
    candidates = []
    if italic:
        candidates += ["/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed-Italic.ttf",
                       "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"]
    if bold:
        candidates += ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
    candidates += ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                   "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

def process_signature(upload, typed, canvas_size, threshold, rotate_auto, thickness):
    W, H = canvas_size
    if upload is not None:
        sig = Image.open(upload)
        sig = ImageOps.exif_transpose(sig).convert("RGBA")
        # Estimate paper background from image corners and make near-white pixels transparent.
        arr = np.array(sig).astype(np.int16)
        rgb = arr[:, :, :3]
        # Relative darkness; preserve dark blue/black ink, remove white/gray paper.
        gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])
        # threshold slider 80..245; low values keep lighter strokes, high removes more background.
        ink = np.clip((threshold - gray) / max(1, threshold - 30) * 255, 0, 255).astype(np.uint8)
        # Reduce color tint from paper; use black ink for clean transparent signature.
        alpha = Image.fromarray(ink, "L").filter(ImageFilter.GaussianBlur(0.35))
        sig = Image.new("RGBA", sig.size, (20, 20, 20, 0))
        sig.putalpha(alpha)
        bbox = sig.getchannel("A").getbbox()
        if bbox:
            sig = sig.crop(bbox)
        if rotate_auto and sig.width > 0 and sig.height > 0:
            # Principal-axis estimate from alpha pixels; rotate only modest skew.
            a = np.array(sig.getchannel("A"))
            yy, xx = np.where(a > 50)
            if len(xx) > 100:
                cov = np.cov(np.vstack((xx, yy)))
                vals, vecs = np.linalg.eigh(cov)
                vx, vy = vecs[:, np.argmax(vals)]
                angle = math.degrees(math.atan2(vy, vx))
                if angle > 90: angle -= 180
                if angle < -90: angle += 180
                if abs(angle) <= 35:
                    sig = sig.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    else:
        if not typed.strip():
            return None
        sig = Image.new("RGBA", (1100, 240), (255, 255, 255, 0))
        d = ImageDraw.Draw(sig)
        font = find_font(155, italic=True)
        d.text((25, 12), typed.strip(), font=font, fill=(10, 10, 10, 255), stroke_width=0)
        bbox = sig.getchannel("A").getbbox()
        if bbox:
            sig = sig.crop(bbox)
    if sig.width < 1 or sig.height < 1:
        return None
    # Bolden strokes a little using max filter.
    if thickness > 0:
        alpha = sig.getchannel("A").filter(ImageFilter.MaxFilter(3 if thickness == 1 else 5))
        sig.putalpha(alpha)
    # Keep signature aspect ratio; caller positions it.
    return sig

st.title("📸 Passport Studio HD")
st.caption("Photo background cleanup, blue studio background, white border and signature overlay.")
st.info("For best results, upload a clear, front-facing photo. Automatic straightening is an estimate; review the preview. This app does not change facial identity.")

uploaded = st.file_uploader("1) Upload photo", type=["jpg", "jpeg", "png", "webp"])
if uploaded:
    original = open_image(uploaded)
    st.image(original, caption="Original upload", use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        straighten = st.checkbox("Auto-straighten face (experimental)", value=True)
        do_remove = st.checkbox("Remove background automatically", value=True)
        bg_name = st.selectbox("Background", ["Sky blue", "Soft light blue", "White", "Off-white"], index=0)
        size_name = st.selectbox("Output size", ["35 × 45 mm (600 × 800 px)", "2 × 2 inch (600 × 600 px)", "Custom 600 × 800 px"], index=0)
    with c2:
        strength = st.slider("HD enhancement strength", 0, 3, 2)
        border = st.slider("White border (px)", 2, 24, 8)
        shadow = st.checkbox("Soft shadow", value=False)
        st.caption("Enhancement uses contrast, color and sharpening. It is not AI face reconstruction like Remini.")
    st.subheader("2) Signature")
    sig_upload = st.file_uploader("Upload signature image (paper background is okay)", type=["jpg", "jpeg", "png", "webp"], key="sig")
    typed_sig = st.text_input("Or type a name/signature", placeholder="Type name here")
    sc1, sc2 = st.columns(2)
    with sc1:
        sig_position = st.selectbox("Signature placement", ["Bottom center", "Bottom left", "Bottom right"], index=0)
        sig_width = st.slider("Signature width (% of photo)", 15, 55, 32)
    with sc2:
        sig_threshold = st.slider("Remove paper background (signature)", 80, 245, 220)
        sig_bold = st.selectbox("Signature stroke", ["Normal", "Bold", "Extra bold"], index=1)
        sig_auto = st.checkbox("Auto-straighten uploaded signature", value=True)
    if st.button("✨ Generate HD passport photo", type="primary", use_container_width=True):
        with st.spinner("Processing photo…"):
            work = original.copy()
            notes = []
            if straighten:
                work, n = auto_straighten(work)
                notes.append(n)
            if do_remove:
                rgba, err = remove_background(work)
                if rgba is not None:
                    bg_map = {"Sky blue": (105, 190, 235), "Soft light blue": (150, 205, 235), "White": (255, 255, 255), "Off-white": (248, 247, 242)}
                    out_size = (600, 800) if "600 × 800" in size_name else (600, 600)
                    work = fit_subject_on_background(rgba, bg_map[bg_name], out_size, margin=.06)
                    notes.append("Background removed and replaced with a soft studio gradient.")
                else:
                    st.error("Background removal failed. Try again, or turn off background removal. Details: " + err)
                    st.stop()
            else:
                out_size = (600, 800) if "600 × 800" in size_name else (600, 600)
                work = ImageOps.fit(work, out_size, method=Image.Resampling.LANCZOS, centering=(.5, .42))
                notes.append("Original background kept.")
            out_size = (600, 800) if "600 × 800" in size_name else (600, 600)
            if work.size != out_size:
                work = ImageOps.fit(work, out_size, method=Image.Resampling.LANCZOS, centering=(.5, .43))
            work = enhance_photo(work, strength)
            # Compose border and optional subtle shadow.
            if shadow:
                frame = Image.new("RGB", (work.width + 2*border + 28, work.height + 2*border + 28), (235, 235, 235))
                inner = Image.new("RGB", (work.width + 2*border, work.height + 2*border), "white")
                inner.paste(work, (border, border))
                shade = Image.new("RGBA", inner.size, (0,0,0,0))
                frame_rgba = frame.convert("RGBA")
                sh = Image.new("RGBA", inner.size, (0,0,0,75)).filter(ImageFilter.GaussianBlur(10))
                frame_rgba.alpha_composite(sh, (18,18))
                frame_rgba.alpha_composite(inner.convert("RGBA"), (0,0))
                framed = frame_rgba.convert("RGB")
            else:
                framed = Image.new("RGB", (work.width + 2*border, work.height + 2*border), "white")
                framed.paste(work, (border, border))
            sig = process_signature(sig_upload, typed_sig, out_size, sig_threshold, sig_auto, 0 if sig_bold == "Normal" else (1 if sig_bold == "Bold" else 2))
            if sig is not None:
                # Add a dedicated white signature strip below the photo, like the reference layout.
                band_h = max(100, int(work.width * 0.20))
                canvas = Image.new("RGB", (framed.width, framed.height + band_h), "white")
                canvas.paste(framed, (0, 0))
                target_w = int(work.width * sig_width / 100)
                target_h = max(1, int(sig.height * target_w / sig.width))
                # Keep signature fully inside the lower white strip.
                target_h = min(target_h, band_h - 24)
                target_w = max(1, int(sig.width * target_h / sig.height))
                sig = sig.resize((target_w, target_h), Image.Resampling.LANCZOS)
                if sig_position == "Bottom center":
                    x = (canvas.width - sig.width) // 2
                elif sig_position == "Bottom right":
                    x = canvas.width - sig.width - 18
                else:
                    x = 18
                y = framed.height + (band_h - sig.height) // 2
                canvas_rgba = canvas.convert("RGBA")
                canvas_rgba.alpha_composite(sig, (max(0, x), max(framed.height, y)))
                framed = canvas_rgba.convert("RGB")
            st.success("Photo generated. Check that the face, background and signature look correct.")
            for note in notes:
                st.caption("• " + note)
            st.image(framed, caption="Generated preview", use_container_width=True)
            buffer = io.BytesIO()
            framed.save(buffer, format="PNG", optimize=True, dpi=(300, 300))
            st.download_button("⬇️ Download HD PNG", data=buffer.getvalue(), file_name="passport_photo_hd.png", mime="image/png", use_container_width=True)
else:
    st.markdown("### What this app does")
    st.markdown("- Passport-style 600×800 or square output\n- Optional background removal and plain background replacement\n- Conservative sharpness, contrast and color enhancement\n- White border and optional soft shadow\n- Signature upload with paper-background cleanup, or typed name overlay")
