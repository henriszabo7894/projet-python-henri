import cv2

class ArtOverlay:
    def __init__(self):
        pass

    def apply_overlay(self, frame, morphed_face, face_box):
        """
        Replace la zone du visage dans la frame originale par le morphing.
        """
        x, y, w, h = face_box
        overlayed = frame.copy()
        morphed_bgr = cv2.cvtColor(morphed_face, cv2.COLOR_RGB2BGR)
        overlayed[y:y+h, x:x+w] = cv2.resize(morphed_bgr, (w, h))
        return overlayed
