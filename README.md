# 📸 Passport Studio HD

A Python & Streamlit based web application to generate professional passport size photographs. It uses AI to remove backgrounds, correctly frames the subject on a soft light-blue studio gradient, and adds a clean white border with an optional bottom signature strip.

## ✨ Features
- **Auto-Straighten:** Corrects slight head tilts automatically using OpenCV Haar Cascades.
- **AI Background Removal:** Removes original messy backgrounds using `rembg` and overlays the subject cleanly.
- **Professional Layout:** Centers head and shoulders on a realistic light-blue studio gradient.
- **Conservative Enhancement:** Natural HD enhancement (White Balance, Contrast, Sharpness) without changing the user's real facial identity or creating a "plastic" look.
- **Signature Strip:** Automatically cleans up uploaded signatures (removes paper background) and integrates it into a neat white strip below the photo.
- **HD Download:** One-click download as a high-quality PNG.

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/bhanulala123-star/passport-studio.git](https://github.com/bhanulala123-star/passport-studio.git)
   cd passport-studio
