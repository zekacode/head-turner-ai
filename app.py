# app.py

import streamlit as st
import os
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from PIL import Image
from dotenv import load_dotenv

# --- PERBAIKAN PENTING DI SINI ---
# Kita gunakan library standard yang stabil agar 'configure' berfungsi
import google.generativeai as genai

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# Configure the Google Generative AI client
try:
    # Ambil API Key dari .env
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API Key not found in .env file.")
    
    # Konfigurasi client (ini hanya jalan jika import google.generativeai as genai)
    genai.configure(api_key=api_key)
    
except (ValueError, AttributeError) as e:
    st.error(f"Error configuring the Google API Key: {e}. Please ensure your .env file is correct.")
    st.stop()

# Set page configuration for Streamlit
st.set_page_config(
    page_title="AI Head Turner",
    page_icon="🤖",
    layout="centered"
)

# --- Helper Functions ---

def create_sphere_visualizer(h_angle, v_angle):
    """
    Creates a dynamic, 3D-like globe with curved lines of longitude/latitude
    that are anchored to the edges and intersect at the handle.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    sphere_outline = plt.Circle((0, 0), 1, color='#3B82F6', fill=False, linewidth=2, zorder=10)
    ax.add_artist(sphere_outline)
    h_rad, v_rad = np.deg2rad(h_angle), np.deg2rad(v_angle)
    px, py = 0.85 * np.sin(h_rad), 0.85 * np.sin(v_rad)
    top, bottom, left, right = (0, 1), (0, -1), (-1, 0), (1, 0)
    start_point = (px, py)
    path_data = [
        (Path.MOVETO, start_point), (Path.CURVE3, (px, 1)), (Path.CURVE3, top),
        (Path.MOVETO, start_point), (Path.CURVE3, (px, -1)), (Path.CURVE3, bottom),
        (Path.MOVETO, start_point), (Path.CURVE3, (-1, py)), (Path.CURVE3, left),
        (Path.MOVETO, start_point), (Path.CURVE3, (1, py)), (Path.CURVE3, right),
    ]
    codes, verts = zip(*path_data)
    path = Path(verts, codes)
    patch = PathPatch(path, facecolor='none', edgecolor='#3B82F6', linestyle='--', alpha=0.8, zorder=1)
    ax.add_patch(patch)
    handle = plt.Circle((px, py), 0.05, color='white', zorder=6)
    ax.add_artist(handle)
    center_dot = plt.Circle((0, 0), 0.03, color='white', zorder=7)
    ax.add_artist(center_dot)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    return fig

# --- AI Generation Function (with Caching) ---
@st.cache_data(ttl=3600) 
def generate_new_pose(_image_bytes, h_angle, v_angle):
    """
    Calls the Gemini AI model to regenerate the image with a new head pose.
    """
    # USE THIS MODEL NAME. It is stable and has the highest free quota.
    # 'gemini-2.5' does not exist in the public API yet. 
    # 'gemini-2.0-flash-exp' is the latest experimental version.
    model_name = 'gemini-1.5-flash' 
    
    try:
        model = genai.GenerativeModel(model_name)
        input_image = Image.open(io.BytesIO(_image_bytes))

        h_direction = "forward"
        if h_angle > 5: h_direction = f"{abs(h_angle)} degrees to the subject's right"
        elif h_angle < -5: h_direction = f"{abs(h_angle)} degrees to the subject's left"

        v_direction = "level"
        if v_angle > 5: v_direction = f"{abs(v_angle)} degrees upward"
        elif v_angle < -5: v_direction = f"{abs(v_angle)} degrees downward"

        prompt = f"""
        You are an expert AI photo editor. Your task is to regenerate the provided image, changing only the head pose of the main subject.

        **Instructions:**
        1.  Analyze the original image to understand the subject's face, features, lighting, and background.
        2.  Regenerate the image, adjusting the subject's head to be turned {h_direction} and tilted {v_direction}.
        
        **Strict Rules:**
        -   **Preserve Identity:** The subject's facial identity, features, hair, and expression must be perfectly preserved.
        -   **Maintain Consistency:** The background, clothing, lighting, shadows, and overall image style must remain identical.
        -   **Single Change Only:** Do not add, remove, or alter any other elements. Only the head pose should change.
        -   **Output:** The output must be only the final image file. Do not output any text or markdown.
        """
        
        response = model.generate_content([prompt, input_image], stream=False)
        
        if response.parts:
            image_data = response.parts[0].data
            return Image.open(io.BytesIO(image_data))
        else:
            st.error("AI refused to generate. Try a different pose.")
            return None

    except Exception as e:
        # If you get a 429 error here, it means you must wait 60 seconds 
        # because the Free Tier limit was reached.
        st.error(f"AI Error: {e}")
        return None
    
# --- Main Application UI ---
st.title("🤖 AI Head Turner Tool")
st.write("Upload a photo and use the sliders to change the subject's head direction.")

st.header("1. Upload Your Photo")
uploaded_file = st.file_uploader("Choose a clear, front-facing portrait...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    original_image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(original_image, use_column_width=True)
        
    with col2:
        st.subheader("2. & 3. Adjust Pose")
        h_angle = st.slider("Yaw (Horizontal)", -45, 45, 0)
        v_angle = st.slider("Pitch (Vertical)", -30, 30, 0)
        st.write("Direction Preview:")
        direction_fig = create_sphere_visualizer(h_angle, v_angle)
        st.pyplot(direction_fig)

    st.header("4. Generate Your Image")
    if st.button("Apply New Pose", type="primary"):
        with st.spinner('The AI is working its magic... Please wait.'):
            img_byte_arr = io.BytesIO()
            image_format = original_image.format or 'PNG'
            original_image.save(img_byte_arr, format=image_format)
            image_bytes = img_byte_arr.getvalue()
            
            generated_image = generate_new_pose(image_bytes, h_angle, v_angle)
        
        if generated_image:
            st.success("Generation Complete!")
            st.image(generated_image, caption="Here is your new image!", use_column_width=True)

else:
    st.info("Please upload an image to get started.")