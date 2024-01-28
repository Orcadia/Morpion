import cv2
import pytesseract
import os

# Set the TESSDATA_PREFIX environment variable
os.environ['TESSDATA_PREFIX'] = 'C:/Program Files/Tesseract-OCR/tessdata'

def image_process(img_path):
    # Load the image
    image   = cv2.imread(img_path, 0)

    # binarization of the image
    image   = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Divide the image into 9 cells
    height, width               = image.shape
    cell_height, cell_width     = height // 3, width // 3
    cells                       = [ [ None for _ in range(3) ] for _ in range(3) ]

    for i in range(3):
        for j in range(3):
            cells[i][j] = image[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]

    return cells

def character_recognition(cells):
    # OCR for each cells with tesseract
    board = [ [ None for _ in range(3) ] for _ in range(3) ]

    for i in range(3):
        for j in range(3):
            # OCR
            cell = cells[i][j]
            text = pytesseract.image_to_string(cell, config='--psm 10 --oem 3')     # return the character in the cell
            text = text.strip()                                                     # remove any spaces or newlines

            # Check if there is an 'X' or an 'O' in the cell
            if 'X' in text.upper() or 'x' in text.lower():
                board[i][j] = 'X'
            elif 'O' in text.upper() or 'o' in text.lower() or '0' in text:
                board[i][j] = 'O'
            else:
                board[i][j] = ' '

    return board

def winner(board):
    # Display the row in Python
    for row in board:
        print(row)

    no_winner = 0
    # Check if there is a winner
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != ' ':
            print(f"\nThe winner is '{board[i][0]}'\n")
            no_winner += 1
            break
        if board[0][i] == board[1][i] == board[2][i] != ' ':
            print(f"\nThe winner is '{board[0][i]}'\n")
            no_winner += 1
            break

    if board[0][0] == board[1][1] == board[2][2] != ' ':
        print(f"\nThe winner is '{board[0][0]}'\n")
        no_winner += 1
    elif board[0][2] == board[1][1] == board[2][0] != ' ':
        print(f"\nThe winner is '{board[0][2]}'\n")
        no_winner += 1
    elif(no_winner == 0):
        print("\nNo winner")

    # Check if there is still moves available
    empty_cells = 0
    if no_winner == 0:
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    empty_cells += 1

        if empty_cells != 0:
            print(f"There is still {empty_cells} moves available\n")

# Test a specific image
def full_process(img_path):
    cells = image_process(img_path)
    board = character_recognition(cells)
    winner(board)

# Test all the images in a specific folder
def test_from_folder(folderName):
    for filename in os.listdir(folderName):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            print('filename: ' + filename)
            full_process(folderName + '/' + filename)

test_from_folder('test')