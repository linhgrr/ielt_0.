import tensorflow as tf
from .multihead_attention import MultiHeadAttention
from .position_encoding import positional_encoding
from tensorflow.keras.layers import Dropout, LayerNormalization, Embedding

def FullyConnect(embedding_dim, fully_connected_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(fully_connected_dim, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(embedding_dim)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate=0.1, layernorm_eps=1e-6):
        super(EncoderLayer, self).__init__()

        # Multi-Head Attention layer
        self.mha = MultiHeadAttention(d_model=embedding_dim, num_heads=num_heads)

        # Feed-Forward Network
        self.ffn = FullyConnect(embedding_dim=embedding_dim,
                                fully_connected_dim=fully_connected_dim)

        # Layer Normalization
        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)

        # Dropout layers
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training, mask):
        """
        Forward pass for the Encoder Layer.

        Arguments:
            x: Tensor of shape (batch_size, input_seq_len, embedding_dim)
            training: Boolean, set to True to activate the training mode for dropout layers
            mask: Boolean mask to ensure that padding is not treated as part of the input

        Returns:
            encoder_layer_out: Tensor of shape (batch_size, input_seq_len, embedding_dim)
        """
        # Multi-Head Self-Attention
        self_attn_output = self.mha(q=x, k=x, v=x, mask=mask)  # (batch_size, input_seq_len, embedding_dim)
        self_attn_output = self.dropout1(self_attn_output, training=training)

        # Add & Norm (Residual connection + Layer Normalization)
        mult_attn_output = self.layernorm1(x + self_attn_output)  # (batch_size, input_seq_len, embedding_dim)

        # Feed-Forward Network
        ffn_output = self.ffn(mult_attn_output)  # (batch_size, input_seq_len, embedding_dim)
        ffn_output = self.dropout2(ffn_output, training=training)

        # Add & Norm (Second residual connection + Layer Normalization)
        encoder_layer_out = self.layernorm2(mult_attn_output + ffn_output)  # (batch_size, input_seq_len, embedding_dim)

        return encoder_layer_out

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_encoder_layers, d_model, num_heads, fully_connected_dim, input_vocab_size, maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers

        self.embedding = Embedding(input_vocab_size, self.d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layer = [EncoderLayer(embedding_dim=d_model,
                                       num_heads=num_heads,
                                       fully_connected_dim=fully_connected_dim,
                                       dropout_rate=dropout_rate,
                                       layernorm_eps=layernorm_eps)
                          for _ in range(self.num_encoder_layers)]

        self.dropout = Dropout(dropout_rate)

    def call(self, x, training, mask):
        """
        Forward pass for the Encoder

        Arguments:
            x -- Tensor of shape (batch_size, input_seq_len)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            mask -- Boolean mask to ensure that the padding is not
                    treated as part of the input
        Returns:
            out2 -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
        """

        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model,tf.float32))

        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_encoder_layers):
            x = self.enc_layer[i](x, training=training, mask=mask)

        return x

