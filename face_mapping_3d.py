import cv2
import mediapipe as mp
import numpy as np

# Charger l'image de Mona Lisa
monalisa = cv2.imread("assets/artworks/Mona_Lisa.jpeg")
if monalisa is None:
    raise FileNotFoundError("Image Mona_Lisa.jpeg introuvable dans assets/artworks/")

# Initialisation MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Indices utiles pour les lèvres et les yeux
UPPER_LIP = [13, 14]
LOWER_LIP = [17, 18]
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]

# Capture webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        h, w, _ = frame.shape

        # Exemple : calculer centre des lèvres
        upper_lip = np.mean([[lm.x * w, lm.y * h] for i, lm in enumerate(face_landmarks.landmark) if i in UPPER_LIP], axis=0)
        lower_lip = np.mean([[lm.x * w, lm.y * h] for i, lm in enumerate(face_landmarks.landmark) if i in LOWER_LIP], axis=0)

        lip_center = ((upper_lip + lower_lip) / 2).astype(int)
        lip_height = int(np.linalg.norm(upper_lip - lower_lip))

        # Redimensionner Mona Lisa pour coller sur la bouche
        scale_factor = lip_height / monalisa.shape[0] * 3  # Ajuste 3 pour taille
        overlay = cv2.resize(monalisa, (int(monalisa.shape[1]*scale_factor), int(monalisa.shape[0]*scale_factor)))
        ox, oy = lip_center[0] - overlay.shape[1] // 2, lip_center[1] - overlay.shape[0] // 2

        # Coller l'image sur la frame
        y1, y2 = max(0, oy), min(frame.shape[0], oy + overlay.shape[0])
        x1, x2 = max(0, ox), min(frame.shape[1], ox + overlay.shape[1])

        overlay_roi = overlay[0:y2 - y1, 0:x2 - x1]
        alpha_mask = np.ones_like(overlay_roi, dtype=np.uint8) * 255

        try:
            frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 0.5, overlay_roi, 0.5, 0)
        except:
            pass  # Au cas où l'overlay dépasse l'écran

    cv2.imshow("Face Mapping 3D", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
