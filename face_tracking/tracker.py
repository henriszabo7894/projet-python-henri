import cv2

class FaceTracker:
    def __init__(self):
        """Initialise un tracker KCF d'OpenCV."""
        self.tracker = cv2.legacy.TrackerKCF_create()
        self.initialized = False
        self.last_frame = None

    def update(self, faces, frame):
        """
        Met à jour le suivi des visages.
        Si aucun tracker n’est initialisé, on démarre avec le premier visage détecté.
        """
        self.last_frame = frame

        if not self.initialized and len(faces) > 0:
            x, y, w, h = faces[0]
            self.tracker.init(frame, (x, y, w, h))
            self.initialized = True
            return [faces[0]]

        if self.initialized:
            success, bbox = self.tracker.update(frame)
            if success:
                return [tuple(map(int, bbox))]
        return []
