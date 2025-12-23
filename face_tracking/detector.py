import cv2

class FaceDetector:
    def __init__(self):
        """Charge le détecteur de visage Haar Cascade d'OpenCV."""
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect_faces(self, frame):
        """
        Détecte les visages dans une image.
        :return: liste de rectangles (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        return faces
