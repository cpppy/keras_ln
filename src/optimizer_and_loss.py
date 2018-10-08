import keras
import numpy as np

def define_loss(model):
    # 多分类问题
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 二分类问题
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 均方误差回归问题
    model.compile(optimizer='rmsprop',
                  loss='mse')

    # 自定义评估标准函数
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', mean_pred])


def mean_pred(y_true, y_pred):
    return keras.backend.mean(y_pred)


# if __name__ == "__main__":
#     y_true = np.array([1.0, 1.0, 1.0])
#     y_pred = np.array([0.7, 0.8, 0.9])
#     print(mean_pred(y_true, y_pred))

def build_model():
    # build model

    input_layer = keras.layers.Input(shape=(10, 10, 3))
    conv1_out = keras.layers.Conv2D(36, (1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    pool1_out = keras.layers.GlobalAveragePooling2D()(conv1_out)
    fc1_out = keras.layers.Dense(20, activation='relu')(pool1_out)
    fc2_out = keras.layers.Dense(1)(fc1_out)

    my_model = keras.models.Model(input_layer, fc2_out)
    return my_model


if __name__ == "__main__":

    my_model = build_model()
    print(my_model.summary())
    my_model.compile(optimizer='rmsprop', loss='mse', metrics=['acc'])

    # train data
    x_train = np.random.random((10, 10, 10, 3))
    y_train = np.random.random((10,1))
    my_model.fit(x_train, y_train, batch_size=2, shuffle=True, validation_split=0.1)

    pred_res = my_model.predict(x_train[:3])
    print(pred_res)
