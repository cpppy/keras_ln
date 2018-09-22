import keras
import numpy as np
import os
import h5py



def get_origin_train_data():
    text = 'abcdefghijklmnopqrstuvwxyz'
    SEQLEN = 5
    STEP = 1
    input_chars = []
    label_chars = []
    for i in range(0, len(text) - SEQLEN, STEP):
        input_chars.append(text[i:i + SEQLEN])
        label_chars.append(text[i + SEQLEN])
    return input_chars, label_chars

def get_char_index_dict():
    text = 'abcdefghijklmnopqrstuvwxyz'
    char2index = {}
    for idx, char in enumerate(text):
        char2index[char] = idx
    return char2index

def get_one_hot_code_train_data(input_chars, label_chars):
    char2index = get_char_index_dict()
    X = np.zeros((len(input_chars), 5, 26))
    Y = np.zeros((len(input_chars), 26))
    for i, char_seq in enumerate(input_chars):
        for j, char in enumerate(char_seq):
            X[i, j, char2index[char]] = 1
            Y[i, char2index[label_chars[i]]] = 1
    return X, Y




def build_RNN_model():

    model = keras.models.Sequential()
    model.add(keras.layers.SimpleRNN(128,
                                     return_sequences=False,
                                     input_shape=(5, 26), unroll=True))
    model.add(keras.layers.Dense(26))
    model.add(keras.layers.Activation("softmax"))
    return model



if __name__ == "__main__":

    print(get_char_index_dict())

    origin_x_input, origin_y_label = get_origin_train_data()
    print("origin_input_data: ", len(origin_x_input), len(origin_y_label))

    x_data, y_label = get_one_hot_code_train_data(origin_x_input, origin_y_label)
    print("input_data(one-hot-code): ", x_data.shape, y_label.shape)


    # model
    model = build_RNN_model()

    #weights
    weights_path = "../model/rnn_ln_weights/rnn_ln_weights_1.h5"
    if os.path.exists(weights_path):
        print("find weights file, loading...")
        model.load_weights(weights_path)
    else:
        print("new model, weights file not exist.")


    model.compile(loss="categorical_crossentropy",
                  optimizer="rmsprop",
                  metrics=['acc'])


    model.fit(x_data, y_label,
              batch_size=8,
              verbose=2,
              epochs=20,
              shuffle=True,
              validation_split=0.2,
              callbacks=[]
              )

    model.save_weights(weights_path)

    test_data = x_data[:5]
    test_result = model.predict(test_data)
    text = 'abcdefghijklmnopqrstuvwxyz'
    for pred_code in test_result:
        idx = np.argmax(pred_code)
        print(text[idx])