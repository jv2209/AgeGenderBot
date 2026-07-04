import cv2
import numpy as np

# ==============================
# SET YOUR MODEL FILE PATH
# ==============================

base_path = r"C:\Users\HP\Downloads\AgeGenderBot"

# Face detection model
face_prototxt = base_path + r"\deploy.prototxt"
face_model = base_path + r"\res10_300x300_ssd_iter_140000.caffemodel"

# Age model
age_prototxt = base_path + r"\age_deploy.prototxt"
age_model = base_path + r"\age_net.caffemodel"

# Gender model
gender_prototxt = base_path + r"\gender_deploy.prototxt"
gender_model = base_path + r"\gender_net.caffemodel"

# ==============================
# LOAD MODELS
# ==============================

face_net = cv2.dnn.readNetFromCaffe(face_prototxt, face_model)
age_net = cv2.dnn.readNetFromCaffe(age_prototxt, age_model)
gender_net = cv2.dnn.readNetFromCaffe(gender_prototxt, gender_model)

# Age and gender categories
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_BUCKETS = ['Male', 'Female']

# ==============================
# RUN CAMERA
# ==============================

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Prepare face detection blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    # Loop through faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            # Compute face box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype(int)

            face = frame[y1:y2, x1:x2]

            # Skip if face is too small
            if face.size == 0:
                continue

            # Prepare blob for age & gender
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                              (78.4263377603, 87.7689143744, 114.895847746),
                                              swapRB=False)

            # Predict gender
            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender = GENDER_BUCKETS[gender_preds[0].argmax()]

            # Predict age
            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            age = AGE_BUCKETS[age_preds[0].argmax()]

            label = f"{gender}, {age}"

            # Draw results
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Age-Gender Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

