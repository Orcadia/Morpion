import cv2
import pytesseract
import numpy as np
import os
# Set the TESSDATA_PREFIX environment variable
os.environ['TESSDATA_PREFIX'] = 'C:/Program Files/Tesseract-OCR/tessdata'

# Load the image
image = cv2.imread('test2.jpg', 0)

# Preprocess the image
image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Divide the image into 9 cells
height, width = image.shape
cell_height, cell_width = height // 3, width // 3
cells = [ [ None for _ in range(3) ] for _ in range(3) ]

for i in range(3):
    for j in range(3):
        cells[i][j] = image[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]

# Recognize the character in each cell
board = [ [ None for _ in range(3) ] for _ in range(3) ]

for i in range(3):
    for j in range(3):
        cell = cells[i][j]
        text = pytesseract.image_to_string(cell, config='--psm 10 --oem 3')
        text = text.strip()

        if 'X' in text.upper() or 'x' in text.lower():
            board[i][j] = 'X'
        elif 'O' in text.upper() or 'o' in text.lower() or '0' in text:
            board[i][j] = 'O'
        else:
            board[i][j] = ' '

# Print the Tic Tac Toe board
for row in board:
    print(row)