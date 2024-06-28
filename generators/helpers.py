
class SuperHelper:
   

   def get_model_path():
      base_path = "E:\\Alkaou\Python Projects\\models\\br_models"
      model_name = input("Entrez le nom de votre model : ")
      model_path = f"{base_path}\\{model_name}.keras"
      return model_path