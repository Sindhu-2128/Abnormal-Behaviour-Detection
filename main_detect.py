import cv2
import joblib
from Feature_extraction import extract_features

# Load model
model = joblib.load("svm_model.pkl")  # Change to knn_model.pkl or logreg_model.pkl if needed

video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    features, bbox = extract_features(frame)

    if features is not None and bbox is not None:
        prediction = model.predict([features])[0]
        x, y, w, h = bbox

        if prediction == 1:
            color = (0, 0, 255)
            label = "Abnormal"
        else:
            color = (0, 255, 0)
            label = "Normal"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Abnormal Behavior Detection", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
