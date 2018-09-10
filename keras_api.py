import keras
import numpy as np

def call_layers(data, labels):
    # 这部分返回一个张量
    inputs = keras.layers.Input(shape=(10,))

    # 层的实例是可调用的，它以张量为参数，并且返回一个张量
    x = keras.layers.Dense(10, activation='relu')(inputs)
    x = keras.layers.Dense(10, activation='relu')(x)
    predictions = keras.layers.Dense(10, activation='softmax')(x)

    # 这部分创建了一个包含输入层和三个全连接层的模型
    model = keras.models.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(data, labels)  # 开始训练

if __name__ == "__main__":

    x_data = np.random.random((2, 10))
    labels = np.random.random((2, 10))
    # call_layers(x_data, labels)


    x_input = keras.layers.Input(shape=(10,))
    y = keras.layers.Dense(10, activation="relu")(x_input)
    predictions = keras.layers.Dense(10, activation='softmax')(y)
    model = keras.models.Model(inputs=x_input, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_data, labels)





