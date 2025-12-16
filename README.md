# 🤖 AI Head Turner

An interactive web application built with Streamlit that allows users to change the head direction (yaw and pitch) in a photograph using a generative AI model.

## ✨ Features

-   **File Upload:** Upload any JPG, JPEG, or PNG image.
-   **Interactive Pose Control:** Use intuitive sliders to set the desired horizontal (yaw) and vertical (pitch) angles.
-   **Live Direction Preview:** A dynamic 3D-like sphere visualizes the selected head direction in real-time.
-   **AI-Powered Generation:** Leverages a powerful image-editing model (`meituan-longcat/LongCat-Image-Edit`) hosted on the `fal.ai` serverless platform for fast and reliable results.

## 🛠️ Tech Stack

-   **Framework:** [Streamlit](https://streamlit.io/)
-   **AI Model Hosting:** [fal.ai](https://fal.ai/) via [Hugging Face InferenceClient](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/inference_client#huggingface_hub.InferenceClient)
-   **Core Libraries:** Python, Pillow, Matplotlib, NumPy

---

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+
-   An API Key from [fal.ai](https://fal.ai/)

### 1. Local Setup

**Clone the repository:**
```bash
git clone https://github.com/zekacode/head-turner-ai.git
cd head-turner-ai
```

**Create a virtual environment and activate it:**
```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

**Install the required libraries:**
```bash
pip install -r requirements.txt
```

**Create your environment file:**
Create a file named `.env` in the root of the project and add your `fal.ai` API key:
```env
FALAI_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Run the Streamlit app:**
```bash
streamlit run app.py
```
The application should now be running in your web browser!

### 2. Deploying on Streamlit Community Cloud

1.  **Push your project to a GitHub repository.** Make sure your `.env` file is listed in your `.gitignore` and is **not** on GitHub.
2.  Go to [share.streamlit.io](https://share.streamlit.io/) and sign in.
3.  Click **"New app"** and connect your GitHub repository.
4.  In the **"Advanced settings"** section, go to the **"Secrets"** tab.
5.  Add your `fal.ai` API key as a secret. The format should be:
    ```toml
    FALAI_API_KEY = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    ```
6.  Click **"Deploy!"**. Your application will be live in a few minutes.