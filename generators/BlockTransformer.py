from FeedForward import FeedForward, layers

class Block(layers.Layer):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout
        self.sa = layers.MultiHeadAttention(
            num_heads=n_head, key_dim=n_embd // n_head, dropout=dropout
        )
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()

    def call(self, x):
        attn_output = self.sa(
            self.ln1(x), self.ln1(x), use_causal_mask=True
        )  # use causal mask to ensure each token can only see previous tokens
        x = x + attn_output
        x = x + self.ffwd(self.ln2(x))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_embd": self.n_embd,
            "n_head": self.n_head,
            "dropout": self.dropout,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
