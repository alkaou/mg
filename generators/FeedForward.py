from importer import layers, models

class FeedForward(layers.Layer):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.n_embd = n_embd
        self.dropout = dropout
        self.net = models.Sequential(
            [
                layers.Dense(4 * n_embd, activation="relu"),
                layers.Dense(n_embd),
                layers.Dropout(dropout),
            ]
        )

    def call(self, x):
        return self.net(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_embd": self.n_embd,
            "dropout": self.dropout,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
