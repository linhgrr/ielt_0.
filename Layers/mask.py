import tensorflow as tf

def create_padding_mask(seq):
    """
    Creates a matrix mask for the padding cells

    Arguments:
        seq -- (n, m) matrix

    Returns:
        mask -- (n, 1, 1, m) binary tensor
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    Returns an upper triangular matrix filled with ones

    Arguments:
        size -- matrix size

    Returns:
        mask -- (size, size) tensor
    """
    mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_masks(inp, tar):
    encoder_padding_mask = create_padding_mask(inp)

    # Decoder Padding Mask: Use for global multi head attention for masking encoder output
    decoder_padding_mask = create_padding_mask(inp)

    # Look Ahead Padding Mask
    decoder_look_ahead_mask = create_look_ahead_mask(tar.shape[1])

    # Decoder Padding Mask
    decoder_inp_padding_mask = create_padding_mask(tar)

    # Combine Look Ahead Padding Mask and Decoder Padding Mask
    decoder_look_ahead_mask = tf.maximum(
        decoder_look_ahead_mask, decoder_inp_padding_mask)

    return encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask