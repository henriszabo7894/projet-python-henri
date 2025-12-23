import cv2

class Camera:
    def __init__(self, camera_index=0, width=640, height=480):
        """Initialise la caméra."""
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise ValueError("Impossible d'ouvrir la caméra.")

    def read(self):
        """Capture une image depuis la caméra (compatible main.py)."""
        ret, frame = self.cap.read()
        if not ret:
            print("Erreur : impossible de lire une image de la caméra.")
            return False, None
        return True, frame

    def release(self):
        """Libère les ressources de la caméra."""
        if self.cap.isOpened():
            self.cap.release()
