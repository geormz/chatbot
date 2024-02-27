import os
from flask import Flask, jsonify, request
import json
import re
import tensorflow as tf
import random
import spacy
import numpy as np

# Cargar el modelo de spaCy
nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)

# Leer el dataset
with open('intent.json') as file:
    data = json.load(file)

# Función para preprocesar el texto
def preprocesar_texto(texto):
    # Reemplazar caracteres no alfabéticos, excepto '.', '?', '!' y '\''
    texto = re.sub(r'[^a-zA-Z.?!\']', ' ', texto)
    # Reemplazar múltiples espacios con uno solo
    texto = re.sub(r'[ ]+', ' ', texto)
    # Eliminar espacios al principio y al final
    return texto.strip()

# Procesar los datos de entrada y salida
inputs, targets = [], []
intent_doc = {}

for intent in data['intents']:
    intent_name = intent['intention']
    if 'responses' in intent:
        intent_doc[intent_name] = intent['responses']
    else:
        intent_doc[intent_name] = []  # Si no hay respuestas definidas, usar una lista vacía
    for text in intent['text']:
        inputs.append(preprocesar_texto(text))
        targets.append(intent_name)

# Tokenizar los datos de entrada
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')
tokenizer.fit_on_texts(inputs)
input_sequences = tokenizer.texts_to_sequences(inputs)
max_seq_length = max(len(seq) for seq in input_sequences)
padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_seq_length, padding='pre')

# Crear los datos de salida categóricos
word_to_index = {word: idx for idx, word in enumerate(sorted(set(targets)))}
num_classes = len(word_to_index)
categorical_targets = tf.keras.utils.to_categorical([word_to_index[target] for target in targets], num_classes=num_classes)

# Definir la arquitectura del modelo
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index)+1, 512),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.2)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Resumen del modelo
model.summary()

# Entrenar el modelo
model.fit(padded_inputs, categorical_targets, epochs=20)

# Guardar el modelo entrenado
#model.save('modelo_entrenado.h5'

# Definir la función de respuesta
def responder(sentence):
    # Buscar la intención correspondiente a la frase del usuario
    for intent in data['intents']:
        if preprocesar_texto(sentence) in [preprocesar_texto(text) for text in intent['text']]:
            return random.choice(intent['responses']), intent['intention']
    
    # Si no se encuentra una intención específica, usar el modelo para predecir
    tokens = [tokenizer.word_index.get(word, tokenizer.word_index['<unk>']) for word in preprocesar_texto(sentence).split()]
    padded_tokens = tf.keras.preprocessing.sequence.pad_sequences([tokens], maxlen=max_seq_length, padding='pre')
    prediction = model.predict(padded_tokens)
    predicted_class_idx = np.argmax(prediction)
    predicted_intent = list(word_to_index.keys())[predicted_class_idx]
    return random.choice(intent_doc[predicted_intent]), predicted_intent

# Ruta para recibir las solicitudes POST
@app.route('/bot', methods=['POST'])
def chatbot_response():
    user_input = request.json['message']
    response_text, intent = responder(user_input)
    return jsonify({'response': response_text, 'intent': intent})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
