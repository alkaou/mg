from importer import os, tf, keras, register_keras_serializable, layers, optimizers
from BlockTransformer import Block
from params import block_size, vocab_size, n_embd, n_head, n_layer, batch_size, learning_rate


# Bigram Language Model
@register_keras_serializable()
class BigramLanguageModel(keras.Model):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_embedding_table = layers.Embedding(vocab_size, n_embd)
        self.position_embedding_table = layers.Embedding(block_size, n_embd)
        self.blocks = [Block(n_embd, n_head) for _ in range(n_layer)]
        self.ln_f = layers.LayerNormalization()
        self.lm_head = layers.Dense(vocab_size)

    def call(self, idx, targets=None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"Input sequence length {T} exceeds block size {self.block_size}")

        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            tf.range(T)[tf.newaxis, :]
        )  # (1,T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            return logits, None

        logits_flat = tf.reshape(logits, [-1, logits.shape[-1]])
        targets_flat = tf.reshape(targets, [-1])
        loss = keras.losses.sparse_categorical_crossentropy(
            targets_flat, logits_flat, from_logits=True
        )
        return logits, tf.reduce_mean(loss)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits, loss = self(x, y)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}

    def test_step(self, data):
        x, y = data
        logits, loss = self(x, y)
        return {"loss": loss}

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            idx_next = tf.random.categorical(logits, num_samples=1)
            idx = tf.concat([idx, idx_next], axis=1)
        return idx

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "n_embd": self.token_embedding_table.output_dim,
            "block_size": self.block_size,
            "n_head": self.blocks[0].sa.num_heads,
            "n_layer": len(self.blocks),
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def create_model():
    # Initialize the model train and plotting loss curves
    model = BigramLanguageModel(vocab_size=vocab_size, n_embd=n_embd, block_size=block_size, n_head=n_head, n_layer=n_layer)
    # print the number of parameters in the model
    model.build((batch_size, block_size))
    print("Number of trainable parameters:", model.count_params())

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate))

    return model


def load_model(model_name):
    model = keras.models.load_model(model_name)
    return model

def load_or_create_model(model_name):
    if os.path.exists(model_name):
        print("Chargement du model en cours...")
        model = load_model(model_name=model_name)
        print("Le model est chargé avec succès !")
    else:
        print("Création du model en cours...")
        model = create_model()
        print("Le model est crée avec succès !")
    return model