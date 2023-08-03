# app_updated.py

import streamlit as st
from PIL import Image
import cv2
import mediapipe as mp
from deepface import DeepFace
import pickle
import numpy as np
from PIL import JpegImagePlugin  # Import JpegImagePlugin from PIL

mp_face_detection = mp.solutions.face_detection

def detect_faces_in_image(image):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(image, bbox, (0, 255, 0), 2)
    return image

def save_faces_dict(faces_dict, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(faces_dict, f)

def load_faces_dict(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def main():
    st.title("FaceNet512 Photos Collection and Recognition")

    faces_dict = {}
    dict_file_path = "faces_dict.pkl"

    # Check if the faces dictionary file exists and load it if available
    try:
        faces_dict = load_faces_dict(dict_file_path)
    except FileNotFoundError:
        st.warning("No existing faces dictionary found. Start adding new faces.")

    # Add new faces and names to the dictionary
    name = st.text_input("Enter the name of the person:")
    photo_files = st.file_uploader("Upload photos", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

    if st.button("Process Photos") and name and photo_files:
        if name not in faces_dict:
            faces_dict[name] = []
        for photo_file in photo_files:
            img = Image.open(photo_file)
            # st.image(img, caption=photo_file.name, use_column_width=True)
            faces_dict[name].append(img)

        # Save the updated faces dictionary as a pickle file
        save_faces_dict(faces_dict, dict_file_path)

        st.write("Name:", name)
        st.write("Number of photos:", len(faces_dict[name]))

    # Face recognition from webcam
    if st.button("Start Face Recognition"):
        cap = cv2.VideoCapture(0)
        placeholder = st.empty()
        if not cap.isOpened():
            st.error("Unable to access the webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = detect_faces_in_image(frame)

            # FaceNet512 will be slow when matching, so we'll use DeepFace to compare embeddings.
            # You can replace this with your own implementation if needed.
            if faces_dict:
                for name, face_list in faces_dict.items():
                    for i, face in enumerate(face_list):
                        # Ensure that the face is in the proper PIL format (JpegImagePlugin.JpegImageFile)
                        if not isinstance(face, JpegImagePlugin.JpegImageFile):
                            raise ValueError("The face image is not in the expected PIL format.")

                        face_embedding = DeepFace.represent(frame, model_name="Facenet512", enforce_detection=False)
                        result = DeepFace.verify(face_embedding, face, model_name="Facenet512")

                        similarity_threshold = 0.6  # Adjust this threshold as needed
                        if result["distance"] < similarity_threshold:
                            st.write(f"Detected face as {name}")
                            break

            # Display the video feed with detected faces
            placeholder.image(frame, channels="BGR", use_column_width=True)

        cap.release()

if __name__ == "__main__":
    main()
