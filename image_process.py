import cv2
import numpy as np

# Charger l'image
image = cv2.imread('test.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Appliquer un flou gaussien pour réduire le bruit
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Détection de contours avec Canny
edges = cv2.Canny(blurred, 50, 150)

# Appliquer la Transformée de Hough pour détecter les lignes
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

# Dessiner les lignes détectées sur l'image originale
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Afficher l'image avec les lignes détectées
cv2.imshow('Lignes detectees', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Découper chaque cellule de la grille
cell_images = []
for y in range(0, len(grid_lines)-1, 3):  # Choisir l'incrément en fonction de la taille de votre grille
    for x in range(0, len(grid_lines[0])-1, 3):  # Choisir l'incrément en fonction de la taille de votre grille
        cell = gray[y:y+grid_size, x:x+grid_size]
        cell_images.append(cell)

print(cell_images.__sizeof__())