import streamlit as st
from model_helper import predict
from PIL import Image
import io

st.title("Fruit - fresh / spoiled")

uploaded_file = st.file_uploader("Upload the file", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(io.BytesIO(uploaded_file.getbuffer())).convert("RGB")
    st.image(uploaded_file, caption="Uploaded File", use_container_width=True)
    prediction = predict(image)
    st.info(f"Predicted Class: {prediction}")