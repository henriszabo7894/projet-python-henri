import random
import cv2

class MorphingEngine:
    def __init__(self):
        """Initialise le moteur de morphing."""
        pass

    def apply_morph(self, face, artworks):
        """
        Simule un effet de morphing artistique :
        - mélange le visage avec une œuvre d'art aléatoire.
        """
        if not artworks:
            return face

        art = random.choice(artworks)
        art_resized = cv2.resize(art, (face.shape[1], face.shape[0]))
        blended = cv2.addWeighted(face, 0.5, art_resized, 0.5, 0)
        return blended
