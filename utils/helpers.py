import os
import cv2

def load_artworks(path):
    """
    Charge toutes les œuvres d'art d'un dossier donné.
    :param path: chemin du dossier d'œuvres
    :return: liste d'images (numpy arrays)
    """
    artworks = []
    if not os.path.exists(path):
        print(f"⚠️  Dossier '{path}' introuvable.")
        return artworks

    for file in os.listdir(path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(path, file))
            if img is not None:
                artworks.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return artworks
