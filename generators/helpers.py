
class SuperHelper:
   

   def get_model_path():
      base_path = "C:\\Users\\neked\\Desktop\\pythons\\models\\br"
      model_name = input("Entrez le nom de votre model : ")
      model_path = f"{base_path}\\{model_name}.keras"
      return model_path

   def remove_lines_empty_in_txt_file(file_path):
      # Ouvrir le fichier en mode lecture
      with open(file_path, 'r', encoding="utf-8") as fichier:
          lignes = fichier.readlines()

      # Filtrer les lignes vides
      lignes_sans_espaces = [ligne for ligne in lignes if ligne.strip()]

      # Écrire les lignes non vides dans le même fichier
      with open(file_path, 'w', encoding="utf-8") as fichier:
          fichier.writelines(lignes_sans_espaces)

      return True
   
   def model_summary(model):
      model.summary()