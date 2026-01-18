import streamlit as st
import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
import requests
import base64
from io import BytesIO

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Interior Design Generator", layout="wide")

STABILITY_API_KEY = "YOUR_STABILITY_API_KEY"  # optional

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    llm = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base"
    ).to(device)

    return blip_processor, blip_model, tokenizer, llm

blip_processor, blip_model, tokenizer, llm = load_models()

# ---------------- FUNCTIONS ----------------
def analyze_room(image):
    inputs = blip_processor(image, return_tensors="pt").to(device)
    output = blip_model.generate(**inputs, max_length=50)
    return blip_processor.decode(output[0], skip_special_tokens=True)

def generate_design(description, style):
    prompt = f"""
    Room description: {description}
    Interior style: {style}

    Suggest:
    - Furniture
    - Color palette
    - Lighting
    - Decor ideas
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm.generate(**inputs, max_length=250)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate
