import keras
import numpy as np


def inception_model(input_image):
    tower_1 = keras.layers.Conv2D(64, (1, 1), padding='same', activation='relu')(input_image)
    tower_1 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = keras.layers.Conv2D(64, (1, 1), padding='same', activation='relu')(input_image)
    tower_2 = keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_image)
    tower_3 = keras.layers.Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)
    print("tower1 shape: ", tower_1.shape)
    print("tower2 shape: ", tower_2.shape)
    print("tower3 shape: ", tower_3.shape)
    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
    print("output shape: ", output.shape)
    # (256, 256, 64)*3 ---> (256, 768, 64)
    return output


def build_and_train_inception_model():
    model_input = keras.layers.Input(shape=(256, 256, 3))
    model_output = inception_model(model_input)
    model = keras.models.Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer='rmsprop',
                  loss='mae',
                  metrics=['accuracy'])

    x_input = []
    y_label = []
    for i in range(20):
        x_input.append(np.random.random((256, 256, 3)))
        y_label.append(np.random.random((768, 256, 64)))
    x_input = np.array(x_input)
    y_label = np.array(y_label)
    model.fit(x_input, y_label)
    predictions = model.predict(x_input)
    print("output_shape: ", predictions.shape)


def add_layer_test():
    input1 = keras.layers.Input(shape=(16,))
    x1 = keras.layers.Dense(8, activation='relu')(input1)
    input2 = keras.layers.Input(shape=(32,))
    x2 = keras.layers.Dense(8, activation='relu')(input2)
    added = keras.layers.add([x1, x2])

    print("output layer: ", added.shape)  # (?, 8)


if __name__ == "__main__":

    model_input = keras.layers.Input(shape=(256, 256, 3))
    model = inception_model(model_input)

    # build_and_train_inception_model()
    
