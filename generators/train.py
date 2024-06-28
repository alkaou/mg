from model import load_or_create_model, keras
from data import train_data_generator, val_data_generator
from params import eval_interval, eval_iters
from helpers import SuperHelper


model_path = SuperHelper.get_model_path()
fit_time = int(input("Nombre de fois du bouce : "))
max_ecpoch = int(input("Nombre de EPOCHS : "))
model = load_or_create_model(model_path)

print("L' entrainement du model est en cours...")
for ft in range(fit_time):
    # Train the model
    model.fit(
        train_data_generator,
        epochs=max_ecpoch,
        steps_per_epoch=eval_interval,
        validation_data=val_data_generator,
        validation_steps=eval_iters,
    )
    keras.models.save_model(model, model_path)
    print(f"étape {ft+1 } fini sur {fit_time}")

print("L' entrainement du model est terminé !")

