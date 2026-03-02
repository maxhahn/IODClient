import numpy as np

TRUE_PAG = np.array(
    [
        [0, 0, 2, 2, 0],
        [0, 0, 2, 0, 0],
        [2, 1, 0, 2, 2],
        [2, 0, 3, 0, 2],
        [0, 0, 3, 3, 0],
    ]
)

TEST_PAG = np.array(
    [
        [0, 0, 2, 2, 0],
        [0, 0, 1, 0, 0],
        [2, 2, 0, 2, 2],
        [2, 0, 3, 0, 2],
        [0, 0, 3, 3, 0],
    ]
)

shd = np.sum(TRUE_PAG != TEST_PAG).item()
print("SHD:", shd)
