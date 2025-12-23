import cv2

class FaceAligner:
    def __init__(self):
        """Prépare les opérations d'alignement."""
        pass

    def align_face(self, frame, face_box):
        """
        Extrait et redresse la région du visage pour le morphing.
        """
        x, y, w, h = face_box
        face = frame[y:y+h, x:x+w]
        aligned = cv2.resize(face, (256, 256))
        return aligned
