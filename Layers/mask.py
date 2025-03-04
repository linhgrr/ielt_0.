import tensorflow as tf

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    """
    Returns a matrix with ones in the upper triangle (excluding diagonal) to mask future tokens

    Arguments:
        size -- matrix size

    Returns:
        mask -- (size, size) tensor
    """
    # Tạo ma trận tam giác trên (1 ở phía trên đường chéo chính)
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # 1 là mask, 0 là không mask

def create_masks(inp, tar):
    encoder_padding_mask = create_padding_mask(inp)
    decoder_padding_mask = create_padding_mask(inp)

    # Look-ahead mask
    look_ahead_mask = create_look_ahead_mask(tar.shape[1])
    # Mở rộng shape để tương thích với attention: (n, 1, tar_len, tar_len)
    look_ahead_mask = look_ahead_mask[tf.newaxis, tf.newaxis, :, :]

    # Padding mask cho Decoder input
    decoder_inp_padding_mask = create_padding_mask(tar)

    # Kết hợp look-ahead và padding mask
    combined_mask = tf.maximum(look_ahead_mask, decoder_inp_padding_mask)

    return encoder_padding_mask, combined_mask, decoder_padding_mask