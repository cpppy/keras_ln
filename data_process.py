
import keras

import numpy as np

def build_model():
    # 对于具有2个类的单输入模型（二进制分类）：
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(32, activation='relu', input_dim=100))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def build_and_train_model():

    # 对于具有10个类的单输入模型（多分类分类）：

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(32, activation='relu', input_dim=100))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 生成虚拟数据

    data = np.random.random((1000, 100))
    labels = np.random.randint(10, size=(1000, 1))

    # 将标签转换为分类的 one-hot 编码
    one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

    # 训练模型，以 32 个样本为一个 batch 进行迭代
    model.fit(data, one_hot_labels, epochs=10, batch_size=32)

def one_hot_code_ln():
    labels = np.random.randint(10, size=(1000, 1))

    # 将标签转换为分类的 one-hot 编码
    one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

    print("one_hot: ", one_hot_labels[1])

if __name__ == "__main__":


    model = build_model()
    # 生成虚拟数据
    data = np.random.random((1000, 100))
    print("data shape: ", data.shape)
    labels = np.random.randint(2, size=(1000, 1))
    print("labels shape: ", labels.shape)

    # 训练模型，以 32 个样本为一个 batch 进行迭代
    print("training model ... ")
    model.fit(data, labels, epochs=10, batch_size=32)

    print("predict --- ")
    x_test = np.random.random((5, 100))
    pred_results = model.predict(x_test)
    print(pred_results)

    one_hot_code_ln()


