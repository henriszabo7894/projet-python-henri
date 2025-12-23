import cv2

class Display:
    def show_frame(self, window_name, frame):
        """Affiche une frame à l'écran."""
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, bgr)

    def close(self):
        """Ferme toutes les fenêtres."""
        cv2.destroyAllWindows()
