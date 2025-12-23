import cv2
import time
from camera.capture import Camera
from camera.processing import ImageProcessor
from face_tracking.detector import FaceDetector
from face_tracking.tracker import FaceTracker
from art_mapping.morphing import MorphingEngine
from art_mapping.overlay import ArtOverlay
from art_mapping.alignment import FaceAligner
from utils.display import Display
from utils.helpers import load_artworks
import os

def main():
    # Initialisation
    camera = Camera()
    processor = ImageProcessor()
    detector = FaceDetector()
    tracker = FaceTracker()
    aligner = FaceAligner()
    morphing = MorphingEngine()
    overlay = ArtOverlay()
    display = Display()

    # Charger les artworks
    artwork_path = "assets/artworks"
    if not os.path.exists(artwork_path) or not os.listdir(artwork_path):
        print(f"⚠️  Dossier '{artwork_path}' introuvable ou vide. Ajoute des images pour le mapping.")
        artworks = []
    else:
        artworks = load_artworks(artwork_path)

    print("Appuyez sur 'q' pour quitter.")

    prev_time = time.time()

    while True:
        # Capture frame
        ret, frame = camera.read()
        if not ret:
            break

        # Prétraitement
        processed = processor.process_frame(frame)

        # Détection du visage
        faces = detector.detect_faces(processed)

        # Suivi du visage
        tracked_faces = tracker.update(faces, frame)

        # Pour chaque visage détecté/suivi
        for face in tracked_faces:
            aligned_face = aligner.align_face(processed, face)
            if artworks:
                morphed = morphing.apply_morph(aligned_face, artworks)
                processed = overlay.apply_overlay(processed, morphed, face)

        # Calcul FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(processed, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Affichage
        display.show_frame("Face Mapping", processed)

        # Quitter si 'q' pressé
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libération
    camera.release()
    display.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
