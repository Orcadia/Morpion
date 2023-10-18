import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Assurez-vous que les dossiers 'data/X' et 'data/O' existent
os.makedirs('data/X', exist_ok=True)
os.makedirs('data/O', exist_ok=True)

# Générer et enregistrer les images de cercles avec contours
for i in range(100):
    # Créer une image vide avec un fond de couleur aléatoire
    background_color = tuple(np.random.randint(0, 256, 3).tolist())
    circle_image = np.full((100, 100, 3), background_color, dtype=np.uint8)

    # Générer un cercle avec une couleur et une position aléatoires
    circle_color = tuple(np.random.randint(0, 256, 3).tolist())
    center = (np.random.randint(10, 90), np.random.randint(10, 90))
    radius = np.random.randint(10, 40)
    cv2.circle(circle_image, center, radius, circle_color, -1)

    # Enregistrer l'image dans le dossier 'data/O'
    cv2.imwrite(f'data/O/circle_{i}.jpg', circle_image)

# Générer et enregistrer les images de croix avec variations
for i in range(100):
    # Créer une image vide avec un fond de couleur aléatoire
    background_color = tuple(np.random.randint(0, 256, 3).tolist())
    cross_image = np.full((100, 100, 3), background_color, dtype=np.uint8)

    # Générer une croix avec une couleur et une orientation aléatoires
    cross_color = tuple(np.random.randint(0, 256, 3).tolist())
    thickness = np.random.randint(1, 5)

    # Choisir une forme de croix aléatoirement (X, +, autre)
    shape_choice = np.random.randint(0, 3)
    if shape_choice == 0:  # X
        cv2.line(cross_image, (20, 20), (80, 80), cross_color, thickness)
        cv2.line(cross_image, (20, 80), (80, 20), cross_color, thickness)
    elif shape_choice == 1:  # +
        cv2.line(cross_image, (50, 20), (50, 80), cross_color, thickness)
        cv2.line(cross_image, (20, 50), (80, 50), cross_color, thickness)
    else:  # Autre forme de croix
        # Ajoutez ici le code pour générer votre propre forme de croix
        pass

    # Enregistrer l'image dans le dossier 'data/X'
    cv2.imwrite(f'data/X/cross_{i}.jpg', cross_image)

# Charger les images et les étiquettes
data_dir = 'data'
classes = ['X', 'O']

images = []
labels = []

for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    class_label = classes.index(class_name)

    for filename in os.listdir(class_dir):
        img_path = os.path.join(class_dir, filename)
        img = load_img(img_path, target_size=(100, 100))
        img = img_to_array(img)
        img = img / 255.0

        images.append(img)
        labels.append(class_label)

images = np.array(images)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Tester le réseau de neurones avec une image existante
test_image_path = 'test/test5.jpg'
test_image = load_img(test_image_path, target_size=(100, 100))
test_image = img_to_array(test_image)
test_image = test_image / 255.0
test_image = np.expand_dims(test_image, axis=0)

prediction = model.predict(test_image)
predicted_class = np.argmax(prediction)

class_names = ['X', 'O']
print(f'Classe prédite : {class_names[predicted_class]}')
print(f'Probabilités : X={prediction[0,0]}, O={prediction[0,1]}')
