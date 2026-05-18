import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image

class_names = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

IMG_SIZE = 32


def predict_multiple_images(model, folder_path):

    image_files = os.listdir(folder_path)

    for img_file in image_files:

        img_path = os.path.join(folder_path, img_file)

        img = image.load_img(img_path,target_size=(IMG_SIZE, IMG_SIZE))
        
        img_array = image.img_to_array(img)

        img_array = img_array / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction)

        plt.imshow(img)

        plt.title(
            f"Prediction: {class_names[predicted_class]}\n"
            f"Confidence: {confidence:.2f}"
        )

        plt.axis('off')

        plt.show()

        print(f"Image: {img_file}")
        print(f"Predicted Class: {class_names[predicted_class]}")
        print(f"Confidence: {confidence:.2f}\n")
