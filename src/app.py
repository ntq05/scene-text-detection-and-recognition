import streamlit as st
import requests
from PIL import Image, ImageDraw
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Scene Text Detection and Recognition",
    layout="wide"
)

st.title("📖 Scene Text Detection and Recognition")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Recognizing text..."):

        response = requests.post(
            API_URL,
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            },
        )

    if response.status_code == 200:

        results = response.json()["predictions"]

        draw = ImageDraw.Draw(image)

        table = []

        for prediction in results:

            bbox = prediction["bbox"]
            text = prediction["text"]
            confidence = prediction["confidence"]

            draw.rectangle(bbox, outline="red", width=3)

            draw.text(
                (bbox[0], bbox[1] - 20),
                text,
                fill="red"
            )

            table.append({
                "Text": text,
                "Confidence": round(confidence, 3)
            })

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(
                image,
                caption="Prediction",
                use_container_width=True
            )

        with col2:
            st.subheader("Recognized Text")
            st.dataframe(
                pd.DataFrame(table),
                use_container_width=True
            )

    else:
        st.error(response.text)