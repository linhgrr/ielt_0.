import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Computes the scaled dot-product attention.

    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len_q, depth)
        k: Key tensor of shape (batch_size, num_heads, seq_len_k, depth)
        v: Value tensor of shape (batch_size, num_heads, seq_len_v, depth)
        mask: Optional mask tensor (broadcastable to scaled_score shape)

    Returns:
        output: Tensor of shape (batch_size, num_heads, seq_len_q, depth)
        attention_weights: Attention weights tensor of shape 
                           (batch_size, num_heads, seq_len_q, seq_len_k)
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch, num_heads, seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_score = matmul_qk / tf.math.sqrt(dk)  # Chia căn bậc hai dk để tránh gradient exploding

    if mask is not None:
        mask = tf.cast(mask, dtype=tf.float32)  # Đảm bảo mask có kiểu float32
        scaled_score += (mask * -1e9)  # Thêm mask để loại bỏ các vị trí không cần tính

    attention_weights = tf.nn.softmax(scaled_score, axis=-1)  # Tính xác suất softmax
    output = tf.matmul(attention_weights, v)  # Nhân với V để lấy output cuối cùng

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        self.depth = int(d_model / num_heads)  # Ép kiểu thành int để reshape

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth) and transpose."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask=None):
        """
        Args:
            q: Query tensor of shape (batch_size, seq_len_q, d_model)
            k: Key tensor of shape (batch_size, seq_len_k, d_model)
            v: Value tensor of shape (batch_size, seq_len_v, d_model)
            mask: Optional mask tensor (broadcastable to attention scores)

        Returns:
            output: Tensor of shape (batch_size, seq_len_q, d_model)
        """
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention: (batch_size, num_heads, seq_len_q, depth)
        # attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # Transpose back to (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # Reshape to (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output