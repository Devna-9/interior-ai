
import streamlit as st
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Interior Designer",
    layout="centered"
)

client = OpenAI(api_key="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")  # <-- add your API key here

# -------------------------------
# BUDGET LOGIC (Feature Engineering Equivalent)
# -------------------------------
def budget_description(budget):
    if budget == "Low":
        return (
            "low-cost materials, compact furniture, simple decor, "
            "laminate flooring, minimal accessories"
        )
    elif budget == "Medium":
        return (
            "mid-range materials, modular furniture, wooden finishes, "
            "balanced decor, premium lighting"
        )
    else:
        return (
            "luxury materials, marble flooring, custom furniture, "
            "designer decor, premium textures"
        )

# -------------------------------
# PROMPT ENGINEERING (Core Logic)
# -------------------------------
def build_prompt(room, style, color, lighting, budget_desc, furniture_style, material):
    prompt = f"""
    Ultra-realistic {style.lower()} {room.lower()} interior design,
    designed for a {budget_desc}.
    Dominant {color.lower()} color palette with realistic textures.
    {furniture_style} furniture with accurate proportions.
    {lighting.lower()} lighting with natural shadows and reflections.
    Global illumination, soft ambient light.
    Dominant use of {material.lower()} textures and finishes.
    Shot with DSLR camera, 35mm lens, shallow depth of field.
    Photorealistic interior design photography, 8K resolution.
    No people, no text, no watermark.
    """
    return prompt

# -------------------------------
# DALL·E IMAGE GENERATION
# -------------------------------
def generate_image(prompt):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return response.data[0].url

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("AI Interior Designer")
st.write("Generate unique interior designs with DALL·E 3")

# Input fields
room = st.selectbox("Select Room Type",
                    ("Living Room", "Bedroom", "Kitchen",
                     "Bathroom", "Office", "Dining Room"))

style = st.selectbox("Select Style",
                     ("Modern", "Minimalist", "Industrial",
                      "Bohemian", "Scandinavian", "Traditional",
                      "Art Deco", "Rustic", "Coastal", "Mid-Century Modern"))

color = st.selectbox("Select Dominant Color Palette",
                     ("Neutral", "Monochromatic", "Vibrant",
                      "Pastel", "Dark", "Earthy"))

lighting = st.selectbox("Select Lighting Style",
                       ("Bright", "Dim", "Natural", "Accent", "Ambient"))

furniture_style = st.selectbox("Select Furniture Style",
                               ("Modern", "Classic", "Rustic", "Minimalist", "Industrial", "Bohemian", "Vintage", "Transitional", "Shabby Chic", "French Provincial", "Art Deco", "Asian", "Contemporary", "Eclectic", "Farmhouse", "Hollywood Regency", "Mediterranean", "Mission", "Nautical", "Southwestern", "Tropical", "Victorian"))

material = st.selectbox("Select Dominant Material/Texture",
                           ("Wood", "Metal", "Glass", "Concrete", "Brick", "Stone", "Leather", "Fabric", "Marble", "Ceramic", "Rattan", "Velvet"))

budget_options = {
    "Low": "low-cost materials, compact furniture, simple decor",
    "Medium": "mid-range materials, modular furniture, wooden finishes",
    "High": "luxury materials, marble flooring, custom furniture"
}
budget = st.select_slider(
    "Select Budget",
    options=list(budget_options.keys()),
    value="Medium"
)

# Generate button
if st.button("Generate Design"):
    with st.spinner("Generating your interior design..."):
        budget_desc = budget_description(budget)
        prompt = build_prompt(room, style, color, lighting, budget_desc, furniture_style, material)
        image_url = generate_image(prompt)

        st.image(image_url, caption="Your AI-Generated Interior Design", use_column_width=True)
        st.success("Design Generated!")
        st.write(f"**Prompt used:** {prompt}")
