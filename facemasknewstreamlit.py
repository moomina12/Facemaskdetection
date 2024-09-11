#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.layers import TFSMLayer
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the trained model
model = TFSMLayer(r"C:\Users\HP\mask_detector_model", call_endpoint='serve')

# Function to make predictions on a single frame
def detect_and_predict_mask(frame):
    # Resize and preprocess the frame
    face = cv2.resize(frame, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    # Get predictions from the model
    preds = model(face)
    mask_prob, without_mask_prob = preds[0]

    return mask_prob, without_mask_prob

# Streamlit UI
st.title("Face Mask Detection")
st.write("Turn on your camera and check if you're wearing a mask!")

# Use webcam to capture frames
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

# Start video capture
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture frame from camera. Please check your camera connection.", icon="🚨")
        break

    # Convert the frame color from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face mask detection
    mask_prob, without_mask_prob = detect_and_predict_mask(frame)

    # Determine the label and color for display
    label = "Mask" if mask_prob > without_mask_prob else "No Mask"
    color = (0, 255, 0) if label == "Mask" else (255, 0, 0)

    # Display label and confidence on the frame
    cv2.putText(frame, f"{label}: {max(mask_prob, without_mask_prob) * 100:.2f}%", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    
    # Update the image in the Streamlit app
    FRAME_WINDOW.image(frame)

cap.release()

st.write("Stopped")


# In[ ]:




