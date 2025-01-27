import cv2
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from PIL import Image

#model loading
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")  
inWidth = 368
inHeight = 368
threshold = 0.2

# Body parts and pose pairs
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "Relbow": 3, "Rwrist": 4,
    "LShoulder": 5, "Lelbow": 6, "Lwrist": 7, "RHip": 8, "RKnee": 9,
    "Rankle": 10, "LHip": 11, "LKnee": 12, "Lankle": 13, "REye": 14,
    "LEye": 15, "Rear": 16, "Lear": 17, "Background": 18
}

POSE_PARTS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "Relbow"], ["Relbow", "Rwrist"],
    ["LShoulder", "Lelbow"], ["Lelbow", "Lwrist"], ["Neck", "RHip"], ["RHip", "RKnee"],
    ["RKnee", "Rankle"], ["Neck", "LHip"], ["LHip", "LKnee"], ["LKnee", "Lankle"],
    ["Neck", "Nose"], ["Nose", "REye"], ["REye", "Rear"], ["Nose", "LEye"], ["LEye", "Lear"]
]


st.title("Human Pose Estimation ")
st.write("Upload an image, for estimating wiestime the human pose using OpenCV.")


uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])


def posedetector(frame):
    """
    Perform pose detection on the input frame.
    """
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  

    assert len(BODY_PARTS) == out.shape[1], "Mismatch in BODY_PARTS and output shape"

    points = []
    for i in range(len(BODY_PARTS)):
        heatmap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatmap)
        x = int((frameWidth * point[0]) / out.shape[3])
        y = int((frameHeight * point[1]) / out.shape[2])
        points.append((x, y) if conf > threshold else None)

    # Draw skeleton
    for pair in POSE_PARTS:
        partFrom = pair[0]
        partTo = pair[1]

        assert partFrom in BODY_PARTS, f"{partFrom} not found in BODY_PARTS"
        assert partTo in BODY_PARTS, f"{partTo} not found in BODY_PARTS"

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

  
    t, _ = net.getPerfProfile()
    freq = cv2.getTickFrequency() / 1000
    cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    return frame


if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

  
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

   
    output_image = posedetector(image_bgr)

   
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    st.image(output_image_rgb, caption="Pose Estimation Output", use_column_width=True)