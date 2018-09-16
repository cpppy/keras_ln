import keras
import numpy as np


def build_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(64, activation='relu', input_dim=20))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model


def generate_data():
    x_train = np.random.random((1000, 20))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
    x_test = np.random.random((100, 20))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = generate_data()

    model = build_model()

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    model_checkpoint_file_path = '../checkpoint/callback_ln_weights.{epoch:02d}-{loss:.2f}.hdf5'
    # ../ checkpoint / callback_ln_weights.h5
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_checkpoint_file_path,
                                                       monitor='val_loss',
                                                       verbose=0,
                                                       save_best_only=False,
                                                       save_weights_only=False,
                                                       mode='auto',
                                                       period=1)
    terminate_on_nan = keras.callbacks.TerminateOnNaN()

    # progbar_logger = keras.callbacks.ProgbarLogger(count_mode='samples')

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                  min_delta=0.02,
                                  patience=3,
                                  verbose=0,
                                  mode='auto')

    remote_monitor = keras.callbacks.RemoteMonitor(root='http://localhost:9000',
                                                   path='/publish/epoch/end/',
                                                   field='data',
                                                   headers=None)

    hist = model.fit(x_train, y_train,
                     shuffle=True,
                     verbose=1,
                     batch_size=32,
                     epochs=10,
                     callbacks=[model_checkpoint,
                                terminate_on_nan,
                                early_stopping,
                                remote_monitor
                                ]
                     )
    
    model.fit_generator(workers=2)

    pred = model.predict(x_test)

    # print(pred)




















