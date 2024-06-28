"""
prompt,segments
"cite deux pays en Afrique et donne leur capitale","cite deux pays en Afrique[|]donne la capitale de deux pays en Afrique"
"nomme trois fruits et leur couleur","nomme trois fruits[|]leur couleur"

"""
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Charger les données
segmentation_data = pd.read_csv('segmentation_data.csv')

# Séparer les prompts et les segments
prompts = segmentation_data['prompt'].values
segments = segmentation_data['segments'].apply(ast.literal_eval).values

# Tokenizer et séquences
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(prompts)
sequences = tokenizer.texts_to_sequences(prompts)
padded_sequences = pad_sequences(sequences, maxlen=50)

# Encode segments as sequences
segment_sequences = [tokenizer.texts_to_sequences(segment) for segment in segments]
padded_segment_sequences = [pad_sequences(segment, maxlen=50) for segment in segment_sequences]

# Flatten the segment sequences and create a single training set
X = []
y = []
for i in range(len(padded_sequences)):
    for segment in padded_segment_sequences[i]:
        X.append(padded_sequences[i])
        y.append(segment)

X = pad_sequences(X, maxlen=50)
y = pad_sequences(y, maxlen=50)

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


"""
Entrainement du model
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Créer le modèle de segmentation
segmentation_model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=50),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(50, activation='softmax')
])

segmentation_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
segmentation_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

"""
Utilisation du model segmenteur
"""

def segment_prompt(prompt):
    input_data = prepare_input(prompt)
    segments = segmentation_model.predict(input_data)
    segmented_prompts = []
    
    for segment in segments:
        segment_text = tokenizer.sequences_to_texts([segment])[0]
        segmented_prompts.append(segment_text)
    
    return segmented_prompts

# Exemple d'utilisation
prompt = "cite deux pays en Afrique et donne leur capitale"
segmented_prompts = segment_prompt(prompt)
print(segmented_prompts)
