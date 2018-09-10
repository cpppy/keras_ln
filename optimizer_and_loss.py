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


if __name__ == "__main__":
    y_true = np.array([1.0, 1.0, 1.0])
    y_pred = np.array([0.7, 0.8, 0.9])
    print(mean_pred(y_true, y_pred))
