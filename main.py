import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

def ear(points):
    A = distance.euclidean(points[1], points[5])
    B = distance.euclidean(points[2], points[4])
    C = distance.euclidean(points[0], points[3])
    return (A + B) / (2.0 * C)

cap = cv2.VideoCapture(0)
mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

blink_count = 0
frames_closed = 0
EAR_THRESHOLD = 0.21

while True:
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mesh.process(rgb)

    if result.multi_face_landmarks:
        face = result.multi_face_landmarks[0]
        h, w, _ = frame.shape

        left = []
        right = []

        for idx in LEFT_EYE:
            p = face.landmark[idx]
            left.append((p.x * w, p.y * h))

        for idx in RIGHT_EYE:
            p = face.landmark[idx]
            right.append((p.x * w, p.y * h))

        avg_ear = (ear(left) + ear(right)) / 2

        if avg_ear < EAR_THRESHOLD:
            frames_closed += 1
        else:
            if frames_closed >= 2:
                blink_count += 1
            frames_closed = 0

        liveness = min(100, blink_count * 20)

        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.putText(frame, f"Liveness Score: {liveness}%", (20,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Advanced Blink Liveness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
