import cv2
import streamlit as st
import numpy as np
from PIL import Image

# Titre et instructions de l'application
st.title("Application de détection de visages")
st.write(
    """
    Cette application utilise l'algorithme Viola-Jones pour détecter des visages dans les images.
    
    **Instructions :**
    1. Chargez une image en utilisant le bouton "Upload Image".
    2. Ajustez les paramètres `scaleFactor` et `minNeighbors` pour optimiser la détection.
    3. Choisissez la couleur des rectangles pour les visages détectés.
    4. Cliquez sur "Détecter les visages" pour voir les résultats.
    5. Vous pouvez enregistrer l'image avec les visages détectés en cliquant sur "Sauvegarder l'image".
    """
)

# Chargement de l'image
uploaded_file = st.file_uploader("Chargez une image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Image chargée", use_column_width=True)

    # Paramètres ajustables pour la détection
    st.sidebar.header("Paramètres de détection")
    scaleFactor = st.sidebar.slider("Facteur d'échelle (scaleFactor)", 1.1, 2.0, 1.1, 0.1)
    minNeighbors = st.sidebar.slider("Nombre minimal de voisins (minNeighbors)", 1, 10, 5)
    rectangle_color = st.sidebar.color_picker("Choisissez la couleur du rectangle", "#FF0000")

    # Bouton de détection des visages
    if st.button("Détecter les visages"):
        # Conversion en niveaux de gris
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

        # Dessin des rectangles autour des visages détectés
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), tuple(int(rectangle_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)), 2)

        st.image(image, caption="Visages détectés", use_column_width=True)
        st.write(f"Nombre de visages détectés : {len(faces)}")

        # Bouton pour sauvegarder l'image
        if st.button("Sauvegarder l'image"):
            output_path = "detected_faces.jpg"
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            with open(output_path, "rb") as file:
                btn = st.download_button(
                    label="Télécharger l'image",
                    data=file,
                    file_name="detected_faces.jpg",
                    mime="image/jpeg",
                )
