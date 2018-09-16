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
    x_train = np.random.random((100, 20))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
    x_test = np.random.random((10, 20))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(10, 1)), num_classes=10)
    return x_train, y_train, x_test, y_test

def generate_batch_data_random(x, y, batch_size):
    """逐步提取batch数据到显存，降低对显存的占用"""
    ylen = len(y)
    loopcount = ylen // batch_size
    while (True):
        i = np.random.randint(0,loopcount)
        print("i = ", i)
        yield x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]

def generate_valid_data(x, y, batch_size):
    """逐步提取batch数据到显存，降低对显存的占用"""
    ylen = len(y)
    loopcount = ylen // batch_size
    while (True):
        i = np.random.randint(0, loopcount)
        yield x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = generate_data()

    model = build_model()

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['acc'])


    # terminate_on_nan = keras.callbacks.TerminateOnNaN()
    #
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=0.02,
                                                   patience=3,
                                                   verbose=1,
                                                   mode='auto')
    #
    # hist = model.fit(x_train, y_train,
    #                  shuffle=True,
    #                  verbose=1,
    #                  batch_size=32,
    #                  epochs=10,
    #                  callbacks=[terminate_on_nan,
    #                             early_stopping
    #                             ]
    #                  )

    # model.fit_generator(workers=2)

    batch_size = 16
    epoch = 3

    train_data_generator = generate_batch_data_random(x_train,
                                                      y_train,
                                                      batch_size=batch_size)
    #
    # for i in range(100):
    #     train_data_generator.__next__()

    model.fit_generator(train_data_generator,
                        steps_per_epoch=len(y_train) // batch_size,   # num of batch in one epoch
                        epochs=epoch,
                        validation_data=generate_valid_data(x_test, y_test, 1),
                        validation_steps=3,    # calc average of 3 valid_batches
                        verbose=1,
                        callbacks=[early_stopping])

    pred = model.predict(x_test)

    # print(pred)




















