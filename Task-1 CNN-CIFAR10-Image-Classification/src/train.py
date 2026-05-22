from tensorflow.keras.callbacks import EarlyStopping

from src.preprocessing import (
    load_dataset,
    data_augmentation
)

from src.model import build_cnn_model


def train_model():

    X_train, y_train, X_test, y_test = load_dataset()

    datagen = data_augmentation()

    model = build_cnn_model()

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)

    history = model.fit(datagen.flow(X_train,y_train,batch_size=32),epochs=35,validation_data=(X_test, y_test),callbacks=[early_stop])

    model.save("models/cnn_cifar10_model.keras")

    return model, history, X_test, y_test
