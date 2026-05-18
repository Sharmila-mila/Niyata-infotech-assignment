from tensorflow.keras import datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_dataset():

    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    X_train = X_train[:30000]
    y_train = y_train[:30000]

    return X_train, y_train, X_test, y_test


def data_augmentation():

    datagen = ImageDataGenerator(rotation_range=15,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True,zoom_range=0.1)
    return datagen
