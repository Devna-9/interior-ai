import streamlit as st
import torch
from PIL import Image
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
import openai
import requests
import base64

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Interior Design Generator", layout="wide")

OPENAI_API_KEY = "YOUR_OPENAI_KEY"
STABILITY_API_KEY = "YOUR_STABILITY_KEY"  # optional

openai.api_key = OPENAI_API_KEY

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, caption_model = load_caption_model()

# ---------------- FUNCTIONS ----------------
def analyze_room(image):
    inputs = processor(image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_design_suggestions(room_desc, style):
    prompt = f"""
    You are an interior designer.
    Room description: {room_desc}
    Design style: {style}

    Give:
    1. Furniture suggestions
    2. Color palette
    3. Lighting ideas
    4. Decor elements
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

def generate_room_image(prompt):
    url = "https://api.stability.ai/v1/generation/stable-diffusion-v1-6/text-to-image"
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "text_prompts": [{"text": prompt}],
        "cfg_scale": 7,
        "height": 512,
        "width": 512,
        "samples": 1,
        "steps": 30
    }
    response = requests.post(url, headers=headers, json=payload)
    image_base64 = response.json()["artifacts"][0]["base64"]
    return Image.open(
        np.array(
            Image.open(
                st.BytesIO(base64.b64decode(image_base64))
            )
        )
    )

# ---------------- UI ----------------
st.title("🏠 AI-Powered Interior Design Generator")

uploaded_file = st.file_uploader("Upload a room image", type=["jpg", "png", "jpeg"])
style = st.selectbox("Choose Design Style", ["Minimal", "Scandinavian", "Modern", "Luxury"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Room", use_column_width=True)

    with st.spinner("Analyzing room..."):
        room_desc = analyze_room(image)

    st.subheader("🧠 Room Analysis")
    st.write(room_desc)

    with st.spinner("Generating design suggestions..."):
        suggestions = generate_design_suggestions(room_desc, style)

    st.subheader("🎨 Interior Design Suggestions")
    st.write(suggestions)

    if st.button("Generate Redesigned Room Image (Optional)"):
        with st.spinner("Generating AI image..."):
            img_prompt = f"{style} interior design of a room. {room_desc}"
            gen_image = generate_room_image(img_prompt)
            st.image(gen_image, caption="AI Redesigned Room")
