import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

# --- PAGE CONFIG ---
# This MUST be the first Streamlit command.
st.set_page_config(
    page_title="What's Cooking? 🥦",
    page_icon="🥦",
    layout="wide",
)

# --- MINIMAL CSS INJECTION ---
# This injects the CSS for styling.
def load_css():
    st.markdown(
        """
        <style>
        /* Main app container */
        [data-testid="stAppViewContainer"] {
            /* This is a fallback, the config.toml sets the main background */
            background-color: #0E1117;
        }

        /* Title */
        h1 {
            color: #4CAF50; /* Green color from your theme */
            text-align: center;
            font-weight: bold;
        }
        
        /* Prediction box */
        .prediction-box {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            font-size: 1.5em; /* 1.5x normal size */
            margin-top: 20px;
        }
        
        .prediction-box strong {
            font-size: 1.7em;
            color: #4CAF50; /* Green */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- MODEL LOADING ---
# @st.cache_resource is the "genius" part. It tells Streamlit to
# load this model ONCE and keep it in memory, so it's not
# re-loading on every user interaction.
@st.cache_resource
def load_yolo_model():
    print("--- Loading OpenVINO Model (This happens only once) ---")
    # We MUST tell it task='classify'
    model = YOLO('best_openvino_model/', task='classify')
    print("--- Model Loaded! ---")
    return model

# --- PREDICTION FUNCTION ---
# A helper function to make predictions and return the result
def predict(model, image):
    # Run prediction
    results = model(image)

    # Get the results
    result = results[0]
    names = result.names
    top1_index = result.probs.top1
    top1_prob = result.probs.top1conf
    
    best_guess_name = names[top1_index]
    confidence = top1_prob.item() * 100
    
    return best_guess_name, confidence

# --- MAIN APP ---
def main():
    # Load our CSS styles
    load_css()
    
    # Load the cached model
    model = load_yolo_model()
    
    st.title("🥦 What's Cooking? 🥦")
    st.write("Upload a photo or use your webcam to classify your food!")

    # --- TABS FOR "CLICK" AND "UPLOAD" ---
    tab1, tab2 = st.tabs(["📁 Upload an Image", "📸 Use Webcam"])

    # --- UPLOAD TAB ---
    with tab1:
        uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            # Open the image using PIL (Python Imaging Library)
            image = Image.open(uploaded_file)
            
            # Display the image
            st.image(image, caption="Your Uploaded Image", use_column_width=True)
            
            # Prediction button
            if st.button("Classify This Image", key="upload_classify"):
                with st.spinner("Classifying..."):
                    # Get prediction
                    guess, conf = predict(model, image)
                    
                    # Display in our styled box
                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            I am <strong>{conf:.2f}%</strong> sure this is:
                            <br>
                            <strong>{guess}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    # --- WEBCAM ("CLICK") TAB ---
    with tab2:
        img_file_buffer = st.camera_input("Click a photo of your food!")

        if img_file_buffer:
            # Open the image from the camera buffer
            image = Image.open(img_file_buffer)
            
            # No need to show the image, st.camera_input does that
            
            # Immediately classify (no button needed for webcam)
            with st.spinner("Classifying..."):
                # Get prediction
                guess, conf = predict(model, image)
                
                # Display in our styled box
                st.markdown(
                    f"""
                    <div class="prediction-box">
                        I am <strong>{conf:.2f}%</strong> sure this is:
                        <br>
                        <strong>{guess}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    # --- CREDITS SECTION (Added) ---
    st.markdown("---") # Horizontal line separator
    st.markdown(
        """
        <div style="text-align: center; font-size: 0.75em; color: #777; margin-top: 30px;">
            Created by: &nbsp;
            <span style="color: #4CAF50;">PALLAVI</span> ❤️ &nbsp; | &nbsp; 
            <span style="color: #4CAF50;">ISHA</span> 🌟 &nbsp; | &nbsp;
            <span style="color: #4CAF50;">ANKIT</span> &nbsp; | &nbsp; 
            <span style="color: #4CAF50;">JEET</span>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()