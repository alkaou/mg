"""
prompt,context
"Quelle est la distance entre la Terre et la Lune?",science
"Comment faire une tarte aux pommes?",cuisine
"Qui a gagné la Coupe du Monde en 2018?",sport
"Qu'est-ce que la théorie de la relativité?",science
"Quelle est la recette du gâteau au chocolat?",cuisine

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Charger les données
data = pd.read_csv('data.csv')

# Séparer les prompts et les contextes
prompts = data['prompt'].values
contexts = data['context'].values

# Encoder les étiquettes de contexte
label_encoder = LabelEncoder()
encoded_contexts = label_encoder.fit_transform(contexts)

# Tokenizer et séquences
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(prompts)
sequences = tokenizer.texts_to_sequences(prompts)
padded_sequences = pad_sequences(sequences, maxlen=50)

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_contexts, test_size=0.2, random_state=42)


"""
Lancement de l'entrainement du model
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Créer le modèle de classification
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=50),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


"""
Comment utiliser les models ensemble

"""

def prepare_input(prompt):
    sequence = tokenizer.texts_to_sequences([prompt])
    padded_sequence = pad_sequences(sequence, maxlen=50)
    return padded_sequence

def classify_context(prompt):
    input_data = prepare_input(prompt)
    predicted_context_index = model.predict(input_data).argmax(axis=-1)[0]
    predicted_context = label_encoder.inverse_transform([predicted_context_index])[0]
    return predicted_context

# Chargement des modèles de langage
model_science = tf.keras.models.load_model('model_science.keras')
model_cuisine = tf.keras.models.load_model('model_cuisine.keras')
model_sport = tf.keras.models.load_model('model_sport.keras')

# Dictionnaire des modèles en fonction du contexte
context_to_model = {
    'science': model_science,
    'cuisine': model_cuisine,
    'sport': model_sport
}

# Fonction pour générer une réponse en fonction du contexte
def generate_response(prompt):
    context = classify_context(prompt)
    selected_model = context_to_model.get(context, None)
    if selected_model is not None:
        input_data = prepare_input(prompt)
        response = selected_model.generate(input_data, max_new_tokens=50)
        return response.numpy()
    else:
        return "Contexte non reconnu."

# Exemple d'utilisation
prompt = "Comment faire une tarte aux pommes?"
response = generate_response(prompt)
print(response)
