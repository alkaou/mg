import os
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from data_list import DATA_LIST

tokenizer_path = "tokenizer.json"
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
trainer = trainers.BpeTrainer(
    vocab_size=20000,
    min_frequency=2,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)


class SuperTokenizer:

    # {files_path_array} doit être un tableau. EX: ["1.txt", "2.txt", "3.txt"]
    @staticmethod
    def fit(files_path_array):
        tokenizer.train(files_path_array, trainer=trainer)
        tokenizer.save(tokenizer_path, pretty=True)
        return True
    
    @staticmethod
    def loader_tokenizer_from_json():
        if os.path.exists(tokenizer_path):
            loader_tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            SuperTokenizer.fit(DATA_LIST)
            loader_tokenizer = Tokenizer.from_file(tokenizer_path)
        return loader_tokenizer
    
    def encode(text):
        loader_tokenizer = SuperTokenizer.loader_tokenizer_from_json()
        encoded = loader_tokenizer.encode(text)
        return encoded.ids
    
    def decode(tokens):
        loader_tokenizer = SuperTokenizer.loader_tokenizer_from_json()
        decoded = loader_tokenizer.decode(tokens)
        if decoded[0] == 220 and decoded[-1] == 220:
            decoded = decoded[1:-1]
        return decoded