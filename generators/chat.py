from importer import np
from Tokenizer import SuperTokenizer
from model import load_or_create_model
from helpers import SuperHelper


md_name = SuperHelper.get_model_path()
model = load_or_create_model(model_name=md_name)

# Generate text
while True:
    prompt = input("Entrez votre prompt : ")
    context = SuperTokenizer.encode(prompt)
    context = np.array([context], dtype=np.int64)
    # print(context)

    generated = model.generate(context, max_new_tokens=50)
    print(SuperTokenizer.decode(generated[0].numpy().tolist()))