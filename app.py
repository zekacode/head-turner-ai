# app.py
#
# An AI-powered web application that allows users to change the head pose
# in a photograph using an interactive UI and a generative AI model.
#
# Author: Putrawin Adha Muzakki
# Tech Stack: Python, Streamlit, Hugging Face, fal.ai

import streamlit as st
import os
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import traceback

# --- 1. CONFIGURATION & INITIALIZATION ---

# Load environment variables from a .env file for local development.
# This line should be at the top to make variables available.
load_dotenv()

# Configure the Inference Client to use the fal.ai provider.
# fal.ai offers a stable and fast serverless GPU environment for running AI models.
try:
    # --- THIS IS THE FIX ---
    # For local development, we directly get the key from the .env file.
    # The st.secrets method is only used when deploying to the cloud and
    # will cause an error if a .streamlit/secrets.toml file is not found.
    api_key = os.getenv("FALAI_API_KEY")
    
    if not api_key:
        raise ValueError("FALAI_API_KEY not found. Please ensure your .env file is correct and in the same folder as app.py.")
    
    # Initialize the client with the fal.ai provider and the API key.
    # The library uses 'token' as the parameter name for the key.
    client = InferenceClient(provider="fal-ai", token=api_key)
    
except (ValueError, AttributeError) as e:
    st.error(f"Configuration Error: {e}")
    st.stop()

# Set the Streamlit page configuration for a professional look.
st.set_page_config(
    page_title="AI Head Turner",
    page_icon="🤖",
    layout="centered"
)

# --- 2. HELPER & AI FUNCTIONS ---

def create_sphere_visualizer(h_angle: int, v_angle: int) -> plt.Figure:
    """
    Generates a dynamic, 3D-like sphere visualizer using Matplotlib.

    The visualizer shows a sphere with curved grid lines that are "pulled"
    by a handle, representing the selected horizontal and vertical angles.

    Args:
        h_angle (int): The horizontal angle (yaw) in degrees.
        v_angle (int): The vertical angle (pitch) in degrees.

    Returns:
        plt.Figure: A Matplotlib figure object containing the visualizer plot.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    # Main sphere outline
    sphere_outline = plt.Circle((0, 0), 1, color='#3B82F6', fill=False, linewidth=2, zorder=10)
    ax.add_artist(sphere_outline)

    # Calculate handle position based on angles
    h_rad, v_rad = np.deg2rad(h_angle), np.deg2rad(v_angle)
    px, py = 0.85 * np.sin(h_rad), 0.85 * np.sin(v_rad)
    
    # Define anchor points and the handle's start point
    top, bottom, left, right = (0, 1), (0, -1), (-1, 0), (1, 0)
    start_point = (px, py)

    # Create four Bezier curve paths from the handle to the sphere's edges
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

    # Draw the handle and central dot
    handle = plt.Circle((px, py), 0.05, color='white', zorder=6)
    ax.add_artist(handle)
    center_dot = plt.Circle((0, 0), 0.03, color='white', zorder=7)
    ax.add_artist(center_dot)

    # Style the plot
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    return fig

@st.cache_data(ttl=3600) # Cache results for 1 hour to save on API calls
def generate_new_pose(_image: Image.Image, h_angle: int, v_angle: int) -> Image.Image | None:
    """
    Calls the AI model via the fal.ai provider with an engineered prompt.

    Args:
        _image (Image.Image): The original PIL Image object.
        h_angle (int): The target horizontal angle.
        v_angle (int): The target vertical angle.

    Returns:
        Image.Image | None: The generated PIL Image, or None if an error occurred.
    """
    # --- Prompt Engineering ---
    # 1. Define the persona for the AI.
    persona = "You are an expert digital artist specializing in photorealistic edits."

    # 2. Create descriptive, clear instructions based on the slider values.
    # We use a "dead zone" to avoid tiny, noisy changes.
    h_direction = "forward"
    if h_angle > 5: h_direction = f"approximately {abs(h_angle)} degrees to the right"
    elif h_angle < -5: h_direction = f"approximately {abs(h_angle)} degrees to the left"

    v_direction = "level"
    if v_angle > 5: v_direction = f"tilted approximately {abs(v_angle)} degrees upward"
    elif v_angle < -5: v_direction = f"tilted approximately {abs(v_angle)} degrees downward"

    # 3. Define strict negative constraints (what the AI should NOT do).
    negative_constraints = (
        "Do not change the person's identity, facial features, hair, or expression. "
        "Do not alter the background, lighting, or clothing. "
        "Preserve the original photo's style and quality."
    )

    # 4. Combine everything into a final, structured prompt.
    prompt = (
        f"{persona} "
        f"Regenerate this image. Your only task is to change the head pose of the person. "
        f"Make them look {h_direction} and {v_direction}. "
        f"{negative_constraints}"
    )

    try:
        # Call the API with the engineered prompt.
        # The model 'meituan-longcat/LongCat-Image-Edit' is specifically
        # designed for instruction-based image editing.
        generated_image = client.image_to_image(
            image=_image,
            prompt=prompt,
            model="meituan-longcat/LongCat-Image-Edit",
        )
        return generated_image

    except Exception as e:
        # Provide user-friendly error messages and log the full traceback for debugging.
        st.error("An error occurred while communicating with the AI model.")
        # Print to console for developer debugging
        print(f"Error Details: {str(e)}")
        print(traceback.format_exc())
        return None

# --- 3. MAIN APPLICATION UI ---

st.title("🤖 AI Head Turner")
st.write(
    "An interactive tool to change the head direction in a photo. "
    "Powered by generative AI hosted on `fal.ai`."
)

# Component 1: File Uploader
st.header("1. Upload Your Photo")
uploaded_file = st.file_uploader(
    "Choose a clear, front-facing portrait for the best results.",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    original_image = Image.open(uploaded_file)

    # Use columns for a clean side-by-side layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(original_image, use_column_width=True)
        
    with col2:
        st.subheader("2. Adjust Pose")
        
        # Component 2 & 3: Sliders and Interactive Visualizer
        h_angle = st.slider("Yaw (Horizontal)", -45, 45, 0, help="Controls left/right turning of the head.")
        v_angle = st.slider("Pitch (Vertical)", -30, 30, 0, help="Controls up/down tilt of the head.")
        
        st.write("Direction Preview:")
        direction_fig = create_sphere_visualizer(h_angle, v_angle)
        st.pyplot(direction_fig)

    # Component 4: Generate Button and Output Area
    st.header("3. Generate Your Image")
    if st.button("Apply New Pose", type="primary"):
        with st.spinner('Connecting to the AI... This may take a moment.'):
            generated_image = generate_new_pose(original_image, h_angle, v_angle)
        
        if generated_image:
            st.success("Generation Complete!")
            st.image(generated_image, caption="Here is your new image!", use_column_width=True)
        else:
            # Error messages are handled inside the generate_new_pose function
            pass
else:
    st.info("Please upload an image to get started.")