from model import create_model, os, keras
from helpers import SuperHelper

def create_dand_save_model_empty(model_path):

    if os.path.exists(model_path):
        print("Le model existe déjà !")
    else:
        print("Création du model en cours...")
        model = create_model()
        keras.models.save_model(model, model_path)
        print("Model est crée et sauvegarder avec succès !")

model_path = SuperHelper.get_model_path()
create_dand_save_model_empty(model_path)