# contain function to create positional encoding
import numpy as np
import tensorflow as tf

def get_angles(pos, i, d):
    angles = pos/ (np.power(10000, (2 * (i//2)) / np.float32(d)))
    return angles

def positional_encoding(positions, d_model):
    """

    Args:
        positions: (int) max number of positions
        d_model: (int) encoding size

    Returns: matrix of shape (1, positions, d) contain position encodings

    """

    positional_matrix = np.zeros((1, positions, d_model))
    for pos in range(positions):
        for d in range(d_model):
            if d % 2:
                positional_matrix[0, pos, d] = np.sin(get_angles(pos, d, d_model))
            else:
                positional_matrix[0, pos, d] = np.cos(get_angles(pos, d, d_model))

    return tf.cast(positional_matrix, dtype=tf.float32)
