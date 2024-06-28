from Tokenizer import SuperTokenizer
from data_list import DATA_LIST

result = SuperTokenizer.fit(files_path_array=DATA_LIST)
print(result)