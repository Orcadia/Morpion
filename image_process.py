import cv2
import pytesseract
import numpy as np
import os

# Set the TESSDATA_PREFIX environment variable
os.environ['TESSDATA_PREFIX'] = 'C:/Program Files/Tesseract-OCR/tessdata'

# Load the image
image = cv2.imread('test5.png', 0)

# Preprocess the image
image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

cv2.imshow("Morpion", image)
cv2.waitKey(0)

# Divide the image into 9 cells
height, width = image.shape
cell_height, cell_width = height // 3, width // 3
cells = [[None for _ in range(3)] for _ in range(3)]

for i in range(3):
    for j in range(3):
        cells[i][j] = image[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]

# Recognize the character in each cell
board = [[None for _ in range(3)] for _ in range(3)]

for i in range(3):
    for j in range(3):
        cell = cells[i][j]
        cv2.imshow("Case", cell)
        cv2.waitKey(0)
        contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contoursLignesCell = np.zeros((cell_height, cell_width), dtype=np.uint8)
        cv2.drawContours(contoursLignesCell, contours, -1, 255, 2)
        cv2.imshow("Contours des lignes", contoursLignesCell)
        cv2.waitKey(0)

        # text = pytesseract.image_to_string(cell)
        # text = text.strip()

        # if 'X' in text.upper() or 'x' in text.lower():
        #     board[i][j] = 'X'
        # elif 'O' in text.upper() or 'o' in text.lower() or '0' in text:
        #     board[i][j] = 'O'
        # else:
        #     board[i][j] = ' '

# Print the Tic Tac Toe board
for row in board:
    print(row)
