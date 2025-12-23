import cv2

class ImageProcessor:
    def __init__(self):
        """Initialise les paramètres du processeur d'image."""
        pass

    def process_frame(self, frame):
        """
        Prépare l'image pour la détection et le mapping :
        - redimensionnement
        - lissage
        - conversion couleur
        """
        resized = cv2.resize(frame, (640, 480))
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
        return rgb
