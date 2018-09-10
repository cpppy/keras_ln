
import numpy as np


if __name__ == "__main__":

    x = np.random.random(3)

    x = np.random.random((3, 2))
    print(x)  # x size: 3*2, value: [0.0, 1.0)

    x = np.random.randint(low=1, high=4, size=5)
    print(x)   # [3 1 3 2 2]  value: [low, high)

    x = np.random.randint(10, size=5)
    print(x)   # [2 0 2 2 2], default low = 0, 10 as high value
