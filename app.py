import streamlit as st
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import clip
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline

import google.generativeai as genai


# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Interior Design Generator", layout="wide")

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
device = "cpu"


# ---------------- LOAD MODELS ----------------

@st.cache_resource
def load_clip():
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess


@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base",use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained( "Salesforce/blip-image-captioning-base").to(device)
    model.eval()
    return processor, model


@st.cache_resource
def load_sd():
    pipe = StableDiffusionPipeline.from_pretrained( "runwayml/stable-diffusion-v1-5" )
    pipe = pipe.to(device)
    return pipe


# ---------------- FUNCTIONS ----------------

def generate_caption(image, processor, model):
    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=50)
    return processor.decode(output[0], skip_special_tokens=True)


def generate_sd_image(prompt, pipe):
    image = pipe(prompt).images[0]
    return image


def clip_similarity(image, text, clip_model, preprocess):
    image_input = preprocess(image).unsqueeze(0)
    text_input = clip.tokenize([text])

    image_features = clip_model.encode_image(image_input)
    text_features = clip_model.encode_text(text_input)

    similarity = cosine_similarity(image_features.detach().numpy(),text_features.detach().numpy())[0][0]

    return round(similarity * 100, 2)


# ---------------- UI ----------------

st.title("AI-Powered Interior Design Generator")
st.caption("CLIP + BLIP + Gemini + Stable Diffusion")

uploaded_image = st.file_uploader("Upload a room image",type=["jpg", "jpeg", "png"])

if uploaded_image:

    image = Image.open(uploaded_image).convert("RGB")

    clip_model, preprocess = load_clip()
    processor, blip_model = load_blip()
    sd_pipe = load_sd()

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Room")

    caption = generate_caption(image, processor, blip_model)

    base_prompt = (f"Modern interior design. {caption}. " f"High-quality interior photography, realistic lighting." )

    generated_image = generate_sd_image(base_prompt, sd_pipe)

    with col2:
        st.image(generated_image, caption="AI Generated Design")

    st.write("### Room Understanding")
    st.write(caption)

    user_prompt = st.text_area("Modify the design")

    if st.button("Regenerate") and user_prompt:
        final_prompt = base_prompt + " " + user_prompt
        updated_image = generate_sd_image(final_prompt, sd_pipe)

        st.image(updated_image, caption="Updated Design")

        score = clip_similarity(    image, user_prompt, clip_model, preprocess)

        st.metric("CLIP Similarity", f"{score}%")
