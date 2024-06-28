from importer import np, tf
from Tokenizer import SuperTokenizer
from data_list import DATA_FOR_USE_NOW
from params import batch_size, block_size


# print(f"{len(DATA_FOR_USE_NOW)} files")

def text_data_for_train(data_array):
    text_data = ""
    # Read the text file
    for dtext in data_array:
        with open(dtext, "r", encoding="utf-8") as f:
            text_data += f"{f.read()}"

    return text_data

text = text_data_for_train(data_array=DATA_FOR_USE_NOW)

# On peut netoyer le text s'il est nécessaire en créant une méthode de nétoyage


# Train and test splits
data = np.array(SuperTokenizer.encode(text), dtype=np.int64)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
# data

# Data loading
def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = np.random.randint(0, len(data) - block_size, batch_size)
    x = np.stack([data[i : i + block_size] for i in ix])
    y = np.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y

# Prepare train/val dataset
def train_data_generator():
    while True:
        yield get_batch("train")


def val_data_generator():
    while True:
        yield get_batch("val")


# Donnée d'entrainement
train_data_generator = tf.data.Dataset.from_generator(
    train_data_generator,
    output_signature=(
        tf.TensorSpec(shape=(batch_size, block_size), dtype=tf.int64),
        tf.TensorSpec(shape=(batch_size, block_size), dtype=tf.int64),
    ),
)

# Donnée de validation
val_data_generator = tf.data.Dataset.from_generator(
    val_data_generator,
    output_signature=(
        tf.TensorSpec(shape=(batch_size, block_size), dtype=tf.int64),
        tf.TensorSpec(shape=(batch_size, block_size), dtype=tf.int64),
    ),
)