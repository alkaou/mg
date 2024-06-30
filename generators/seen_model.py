from helpers import SuperHelper
from model import load_or_create_model


model_path = SuperHelper.get_model_path()

model = load_or_create_model(model_path)

def seen_model(model):

    SuperHelper.model_summary(model)

    # ask = input("Discutez avec IA ? Y or N : ")
    # if ask != "n" or ask != "N":
    #     from chat import ChatClass
    #     ChatClass.chat.chat(model=model)



seen_model(model)
