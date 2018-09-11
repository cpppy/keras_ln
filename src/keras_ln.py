from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
import numpy
import keras
import h5py


def build_model():
    '''
            第一步：选择模型
        '''
    model = Sequential()
    '''
       第二步：构建网络层
    '''
    model.add(Dense(500, input_shape=(784,)))  # 输入层，28*28=784
    model.add(Activation('tanh'))  # 激活函数是tanh
    model.add(Dropout(0.5))  # 采用50%的dropout

    model.add(Dense(500))  # 隐藏层节点500个
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(10))  # 输出结果是10个类别，所以维度是10
    model.add(Activation('softmax'))  # 最后一层用softmax作为激活函数

    '''
       第三步：编译
    '''
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # 优化函数，设定学习率（lr）等参数
    model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode='categorical')  # 使用交叉熵作为loss函数

    '''
       第四步：训练
       .fit的一些参数
       batch_size：对总的样本数进行分组，每组包含的样本数量
       epochs ：训练次数
       shuffle：是否把数据随机打乱之后再进行训练
       validation_split：拿出百分之多少用来做交叉验证
       verbose：屏显模式 0：不输出  1：输出进度  2：输出每次的训练结果
    '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()  # 使用Keras自带的mnist工具读取数据（第一次需要联网）
    # 由于mist的输入数据维度是(num, 28, 28)，这里需要把后面的维度直接拼起来变成784维
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
    Y_train = (numpy.arange(10) == y_train[:, None]).astype(int)
    Y_test = (numpy.arange(10) == y_test[:, None]).astype(int)

    model.fit(X_train, Y_train, batch_size=200, epochs=50, shuffle=True, verbose=0, validation_split=0.3)
    model.evaluate(X_test, Y_test, batch_size=200, verbose=0)

    '''
        第五步：输出
    '''
    print("test set")
    scores = model.evaluate(X_test, Y_test, batch_size=200, verbose=0)
    print("")
    print("The test loss is %f" % scores)
    result = model.predict(X_test, batch_size=200, verbose=0)

    result_max = numpy.argmax(result, axis=1)
    test_max = numpy.argmax(Y_test, axis=1)

    result_bool = numpy.equal(result_max, test_max)
    true_num = numpy.sum(result_bool)
    print("")
    print("The accuracy of the model is %f" % (true_num / len(result_bool)))


def load_part_of_layer(fname):
    """
    假如原模型为：
        model = Sequential()
        model.add(Dense(2, input_dim=3, name="dense_1"))
        model.add(Dense(3, name="dense_2"))
        ...
        model.save_weights(fname)
    """
    # new model
    model = Sequential()
    model.add(Dense(2, input_dim=3, name="dense_1"))  # will be loaded
    model.add(Dense(10, name="new_dense"))  # will not be loaded

    # load weights from first model; will only affect the first layer, dense_1.
    model.load_weights(fname, by_name=True)

def save_keras_model(model):
    # only save weights, can be used only after model structure defined
    # model.save_weights("path")

    # model structure(not include weights and config)
    json_string = model.to_json()
    yaml_string = model.to_yaml()

    print("structure in json: ", json_string)

    # load from json or yaml
    model = keras.models.model_from_json(json_string)
    model = keras.models.model_from_yaml(yaml_string)

def output_intermediate_value(model):

    intermediate_output = keras.backend.function([model.layers[0].input],
                                                 [model.layers[3].output])
    outputX = intermediate_output([x_test[0:1]])[0]
    print(outputX)

    # if model include dropout layer or bn layer, tag of learning_phase is needed
    intermediate_output = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()],
                                                 [model.layers[0].output])

    output_in_test_model = 0
    output_in_train_model = 1
    outputX = intermediate_output([x_test[0:3], output_in_train_model])[0]
    print(outputX)

def set_trainable(Input, Model, data, labels):
    x = Input(shape=(32,))
    layer = Dense(32)
    layer.trainable = False
    y = layer(x)

    frozen_model = Model(x, y)
    # in the model below, the weights of `layer` will not be updated during training
    frozen_model.compile(optimizer='rmsprop', loss='mse')

    layer.trainable = True
    trainable_model = Model(x, y)
    # with this model the weights of the layer will be updated during training
    # (which will also affect the above model since it uses the same layer instance)
    trainable_model.compile(optimizer='rmsprop', loss='mse')

    frozen_model.fit(data, labels)  # this does NOT update the weights of `layer`
    trainable_model.fit(data, labels)  # this updates the weights of `layer`

def use_h5py_as_database():
    # use hdf5 as database
    with h5py.File('input/file.hdf5', 'r') as f:
        X_data = f['X_data']
        model.predict(X_data)


if __name__ == "__main__":
    a = [[1, 2], [3, 4]]
    row_sum = np.sum(a, axis=0)
    print(row_sum)
    col_sum = np.sum(a, axis=1)
    print(col_sum)

    # keras ln
    x_train = []
    y_train = []
    for i in range(100):
        tempX = np.arange(i, 100 + i, 1)
        x_train.append(tempX)
        ohCode = np.zeros(10)
        ohCode[int(i / 10)] = 1
        y_train.append(ohCode)

    x_test = []
    y_test = []
    for i in range(10):
        tempX = np.arange(i + 100, 200 + i, 1)
        x_test.append(tempX)
        ohCode = np.zeros(10)
        ohCode[i] = 1
        y_test.append(ohCode)

    x_train = np.array(x_train, dtype="float64")
    y_train = np.array(y_train, dtype="float64")
    x_test = np.array(x_test, dtype="float64")
    y_test = np.array(y_test, dtype="float64")

    print("x_train shape: ", x_train.shape)
    print("y_train shape: ", y_train.shape)

    model = Sequential()
    # model.add(Dense(units=64, input_dim=100))
    model.add(Dense(units=64, input_shape=(100, )))
    model.add(Activation("relu"))
    model.add(Dense(units=64))
    model.add(Activation("relu"))
    model.add(Dense(units=64))
    model.add(Activation("relu"))
    model.add(Dense(units=10))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy',
                  optimizer="sgd",
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=100, batch_size=32)
    # model.train_on_batch(x_batch, y_batch)
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=5)
    print(loss_and_metrics)
    classes = model.predict(x_test[:3], batch_size=5, verbose=1)

    print('classes: ', classes)


    # save model (structure, weights, training config, optimizer status)
    model.save("../model/my_keras_model_save.h5")
    # load from h5???
    model = keras.models.load_model("../model/my_keras_model_save.h5")


    # l1 l2 dropout, will be invalid in test model (valid in train model)

    # if data is too large, use small part of data to train
    # fit in little scale
    model.train_on_batch(x_test, y_test)

    # early stopping in train
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
    model.fit(x_train, y_train,
              validation_split=0.2,
              callbacks=[early_stopping])
    # validation_split: decide how much to train and test

    hist = model.fit(x_train, y_train, epochs=10, batch_size=32)
    print("fit history: ", hist.history)

    # reset model
    model.reset_states()
    # model.layers[0].reset_states()

    frozen_layer = Dense(32, trainable=False)
    model.add(frozen_layer)

    # delete last layer
    model.pop()

    # use famous model
    model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)






