from tensorflow.keras import models, layers


def build_cnn_model():

    model = models.Sequential()

    # First CNN Block
    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation='relu',
            padding='same',
            input_shape=(32, 32, 3)
        )
    )

    model.add(layers.BatchNormalization())

    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation='relu',
            padding='same'
        )
    )

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Dropout(0.25))

    # Second CNN Block
    model.add(
        layers.Conv2D(
            64,
            (3, 3),
            activation='relu',
            padding='same'
        )
    )

    model.add(layers.BatchNormalization())

    model.add(
        layers.Conv2D(
            64,
            (3, 3),
            activation='relu',
            padding='same'
        )
    )

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Dropout(0.25))

    # Third CNN Block
    model.add(
        layers.Conv2D(
            128,
            (3, 3),
            activation='relu',
            padding='same'
        )
    )

    model.add(layers.BatchNormalization())

    model.add(
        layers.Conv2D(
            128,
            (3, 3),
            activation='relu',
            padding='same'
        )
    )

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Dropout(0.3))

    # Flatten Layer
    model.add(layers.Flatten())

    # Dense Layers
    model.add(
        layers.Dense(
            256,
            activation='relu'
        )
    )

    model.add(layers.Dropout(0.5))

    # Output Layer
    model.add(
        layers.Dense(
            10,
            activation='softmax'
        )
    )

    return model