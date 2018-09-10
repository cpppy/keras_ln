import keras
import numpy as np

if __name__ == "__main__":
    a = np.random.random((2, 3, 4))
    b = np.random.random((2, 3, 4))

    c = np.concatenate((a, b), axis=-1)

    print(c.shape)
