"""
Name: Sashen Moodley
Student Number: 219006946
"""

import random
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Bidirectional, concatenate, GRU
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
from keras import backend as K
from nltk.translate.bleu_score import SmoothingFunction

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = "fra.txt"

# --------------Plotting Function------------------------
def plot_graphs(history, string, title):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.title(title)
    plt.show()

# ----------Legacy Metric Functions------------------------------
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# ====================== PREPARING THE DATA ============================
# From Keras tutorial
# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")

for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text, _ = line.split("\t")
    # We use "tab" as the "start sequence" character for the targets,
    # and "\n" as "end sequence" character.
    target_text = "\t" + target_text + "\n"
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1:, input_token_index[" "]] = 1.0
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1:, target_token_index[" "]] = 1.0
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0


# ====================== BUILD THE MODEL ========================================
# Define an input sequence and process it.
encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
encoder = Bidirectional(GRU(latent_dim, return_state=True))
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

encoder_states = concatenate([state_h, state_c])

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_gru = keras.layers.GRU(latent_dim*2, return_sequences=True, return_state=True)
decoder_outputs, idk_state_h = decoder_gru(decoder_inputs, initial_state=encoder_states)
decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

# ====================== TRAIN THE MODEL =====================================
model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
              metrics=["accuracy", 'mean_squared_error', f1_m, precision_m, recall_m])

start_time = time.time()
history = model.fit([encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2)
end_time = time.time()
print(f"Total time to train: {end_time-start_time}s")

# Save model
model.save("Architecture2")
model.save("Architecture2.h5")

# Plotting the training and validation metrics
plot_graphs(history, 'accuracy', 'Architecture 2 Accuracy')
plot_graphs(history, 'loss', 'Architecture 2 Loss')
plot_graphs(history, 'mean_squared_error', 'Architecture 2 MSE')
plot_graphs(history, 'f1_m', 'Architecture 2 F-1 Score')
plot_graphs(history, 'precision_m', 'Architecture 2 Precision')
plot_graphs(history, 'recall_m', 'Architecture 2 Recall')

# ===================== RUNNING INFERENCE (SAMPLING) ===========================
# Hybrid adaptation from Keras and Paeperspace tutorial
# Note: The models are not being loaded from the disk, but rather uses the runtime variables

## Encoder
encoder_model = keras.Model(inputs=encoder_inputs, outputs=encoder_states)

## Decoder
# The below tensors will hold the states of the previous time step
decoder_state_input_h = keras.Input(shape=(latent_dim*2))
decoder_hidden_state_input = [decoder_state_input_h]

decoder_outputs, state_h_dec = decoder_gru(decoder_inputs, initial_state=decoder_hidden_state_input)

decoder_states = [state_h_dec]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = keras.Model(inputs=[decoder_inputs] + decoder_hidden_state_input,
                            outputs=[decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # From Keras tutorial
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:  #
        output_tokens, h = decoder_model.predict([target_seq] + [states_value])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h]
    return decoded_sentence

# Getting random numbers for sentences
rand_indices = [random.randint(0, 50) for _ in range(20)]
# Smoothing function for BLEU
smoothing_func = SmoothingFunction()

# Generating decoded sentences:
start_time = time.time()
scores = []
for seq_index in rand_indices:
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print("-")
    print("Input sentence:", input_texts[seq_index])
    print("Decoded sentence:", decoded_sentence)

    # Putting some reference for the BLEU scores
    if 'Jump' in input_texts[seq_index]:
        references = ['Saute.']
    elif 'Begin' in input_texts[seq_index]:
        references = ['Commence.', 'Commencez.']
    elif 'Wait' in input_texts[seq_index]:
        references = ['Attendez.', 'Attends.', 'Attendez !', 'Attends !', 'Attendez.', 'Attendez !', 'Attends !']
    elif 'Go on' in input_texts[seq_index]:
        references = ['Poursuis.', 'Continuez.', 'Poursuivez.']
    elif 'Go' in input_texts[seq_index]:
        references = ['Va !', 'Marche.', 'Bouge !']
    elif 'Duck' in input_texts[seq_index]:
        references = ['À terre !', 'Baisse-toi !', 'Baissez-vous !']
    elif 'Wow' in input_texts[seq_index]:
        references = ['Ça alors !', 'Waouh !', 'Wah !']
    elif 'Run' in input_texts[seq_index]:
        references = ['Cours !', 'Courez !', 'Prenez vos jambes à vos cous !', 'File !', 'Filez !', 'Cours !', 'Fuyez !', 'Fuyons !']
    elif 'Hello' in input_texts[seq_index]:
        references = ['Bonjour !', 'Salut !']
    elif 'Stop' in input_texts[seq_index]:
        references = ['Ça suffit !', 'Stop !', 'Arrête-toi !']
    elif 'Who' in input_texts[seq_index]:
        references = ['Qui ?']
    elif 'Hide' in input_texts[seq_index]:
        references = ['Cache-toi.', 'Cachez-vous.']
    elif 'Hi' in input_texts[seq_index]:
        references = ['Salut !', 'Salut.']
    elif 'Fire' in input_texts[seq_index]:
        references = ['Au feu !']
    else:
        references = input_texts[seq_index]

    score = sentence_bleu(references, decoded_sentence, weights=(0.5, 0.5), smoothing_function=smoothing_func.method1)
    scores.append(score)
    print(f"BLEU Score : {score}")

end_time = time.time()
print(f"Total time to make predictions: {end_time-start_time}s")
print(f"Average BLEU Score: {sum(scores)/len(scores)*100}%")
