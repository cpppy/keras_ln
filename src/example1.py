import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 生成虚拟数据
    import numpy as np

    x_train = np.random.random((1000, 20))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
    x_test = np.random.random((100, 20))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

    model = Sequential()
    # Dense(64) 是一个具有 64 个隐藏神经元的全连接层。
    # 在第一层必须指定所期望的输入数据尺寸：
    # 在这里，是一个 20 维的向量。
    model.add(Dense(64, activation='relu', input_dim=20))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # config SGD parameters
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['mae', 'acc'])

    fit_history = model.fit(x_train, y_train,validation_split=0.1,
                            shuffle=True,
                            callbacks=[],
                            epochs=100,
                            batch_size=32)
    score = model.evaluate(x_test, y_test, batch_size=128)
    print("x_test shape: ", x_test.shape)
    print("score: ", score)
    evaluate_loss = score[0]
    evaluate_acc = score[1]

    print(fit_history.history.keys())
    # summarize history for accuracy
    # plt.plot(fit_history.history['acc'])
    # plt.plot(fit_history.history['val_acc'])
    plt.plot(fit_history.history['loss'])
    plt.plot(fit_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.show()
