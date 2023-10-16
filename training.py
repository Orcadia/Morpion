import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Définir les dimensions des images
img_height, img_width = 100, 100

# 1. Organiser les données (voir étape 1)

# 2. Importer les bibliothèques (voir étape 2)

# 3. Charger les images et les étiquettes (voir étape 3)

data_dir = 'dataset'
classes = ['X', 'O']  # Liste des classes

images = []
labels = []

for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    class_label = classes.index(class_name)

    for filename in os.listdir(class_dir):
        img_path = os.path.join(class_dir, filename)
        img = load_img(img_path, target_size=(img_height, img_width))
        img = img_to_array(img)
        img = img / 255.0  # Normalisation

        images.append(img)
        labels.append(class_label)

# 4. Convertir en tableau NumPy (voir étape 4)
images = np.array(images)
labels = np.array(labels)

# 5. Diviser les données en ensembles d'entraînement et de test (voir étape 5)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 6. Créer le modèle CNN (voir étape 6)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2, activation='softmax')  # 2 classes: X et O
])

# 7. Compiler et entraîner le modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# 8. Évaluer le modèle (voir étape 7)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# 9. Utilisation du modèle pour la prédiction (voir étape 8)
new_image_path = 'test/test4.jpg'
new_image = load_img(new_image_path, target_size=(img_height, img_width))
new_image = img_to_array(new_image)
new_image = new_image / 255.0  # Normalisation
new_image = np.expand_dims(new_image, axis=0)  # Ajouter une dimension pour correspondre à la forme attendue par le modèle

prediction = model.predict(new_image)
predicted_class = np.argmax(prediction)

class_names = ['X', 'O']
print(f'Classe prédite : {class_names[predicted_class]}')
print(f'Probabilités : X={prediction[0,0]}, O={prediction[0,1]}')
