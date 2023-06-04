import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import os
import glob
import nltk
import string
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pymorphy2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

#folder_path = "Dataset/test"
#folder_path2 = "Dataset/test1"
folder_path = "Dataset/plagiarism1"
folder_path2 = "Dataset/source1"

file_list = glob.glob(folder_path + "/*.txt")
file_list2 = glob.glob(folder_path2 + "/*.txt")

triplets = []
tuples_of_plagiarised_text = []
tuples_of_original_text = []
lstm_units = 128
epochs = 2
batch_size = 8
vocab_size = 20000
max_sequence_length = 1000
embedding_dim = 200


for file_path in file_list:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        tuples_of_plagiarised_text.append([text, file_path])

for file_path2 in file_list2:
    with open(file_path2, "r", encoding="utf-8") as file2:
        text = file2.read()
        tuples_of_original_text.append([text, file_path2])

#print(tuples_of_original_text[0])
#print(tuples_of_plagiarised_text[0])

cleaned_tuples_of_plagiarised_text = []
cleaned_tuples_of_original_text = []

def preprocess_text(text, max_sequence_length):
    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))


    tokens = word_tokenize(text)


    stop_words = set(stopwords.words('russian'))
    lemmatized_tokens = [token for token in tokens if token not in stop_words]

    morph = pymorphy2.MorphAnalyzer()
    lemmatized_tokens = [morph.parse(token)[0].normal_form for token in tokens]

    if len(lemmatized_tokens) < max_sequence_length:
        lemmatized_tokens = lemmatized_tokens + [''] * (max_sequence_length - len(lemmatized_tokens))
    else:
        lemmatized_tokens = lemmatized_tokens[:max_sequence_length]

    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text





for i in tuples_of_original_text:
    text = preprocess_text(i[0], max_sequence_length)
    cleaned_tuples_of_original_text.append([text, i[1]])

for i in tuples_of_plagiarised_text:
    text = preprocess_text(i[0], max_sequence_length)
    cleaned_tuples_of_plagiarised_text.append([text, i[1]])


directory = 'C:/PythonProjectsFolder/KerasProject/Dataset/tasks1'

for filename in os.listdir(directory):
    if filename.endswith(".json"):
        file_path = os.path.join(directory, filename)

        with open(file_path, 'r') as file:
            data = json.load(file)

            #print(data)
        plagiarism_ids = [d["id"] for d in data["plagiarism"]]
        suspicious_document = data["suspicious-document"]

        #print("Plagiarism IDs:", plagiarism_ids)
        #print("Suspicious document:", suspicious_document)

        for t in cleaned_tuples_of_plagiarised_text:
            if (t[1] == 'Dataset/plagiarism1\\' + suspicious_document):
                Positive = t[0]
                for d in plagiarism_ids:
                    for i in cleaned_tuples_of_original_text:
                        if (i[1] == 'Dataset/source1\\' + d + '.txt'):
                            Anchor = i[0]
                            randomIndex = random.randrange(len(tuples_of_original_text))
                            Negative = cleaned_tuples_of_original_text[randomIndex][0]
                            triplets.append([Anchor, Positive, Negative])


def create_siamese_model(vocab_size, embedding_dim, lstm_units):
    anchor_input = Input(shape=(None,))

    positive_input = Input(shape=(None,))

    negative_input = Input(shape=(None,))

    embedding_layer = Embedding(vocab_size, embedding_dim)

    lstm_layer = LSTM(lstm_units)

    encoded_anchor = lstm_layer(embedding_layer(anchor_input))
    encoded_positive = lstm_layer(embedding_layer(positive_input))
    encoded_negative = lstm_layer(embedding_layer(negative_input))

    cosine_similarity = Lambda(lambda tensors: K.sum(tensors[0] * tensors[1], axis=-1) / (
                K.sqrt(K.sum(K.square(tensors[0]), axis=-1)) * K.sqrt(K.sum(K.square(tensors[1]), axis=-1))))

    similarity_scores_pos = cosine_similarity([encoded_anchor, encoded_positive])
    similarity_scores_neg = cosine_similarity([encoded_anchor, encoded_negative])

    siamese_model = Model(inputs=[anchor_input, positive_input, negative_input],
                          outputs=[similarity_scores_pos, similarity_scores_neg])

    return siamese_model


def triplet_loss(y_true, y_pred):
    margin = K.constant(0.2)

    pos_similarity = y_pred[0]
    neg_similarity = y_pred[1]

    loss = K.maximum(0.0, margin + neg_similarity - pos_similarity)

    return loss


anchor_text = []
positive_text = []
negative_text = []

for i in triplets:
    anchor_text.append(i[0])
    positive_text.append(i[1])
    negative_text.append(i[2])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(anchor_text + positive_text + negative_text)
vocab_size = len(tokenizer.word_index) + 1

anchor_sequences = tokenizer.texts_to_sequences(anchor_text)
positive_sequences = tokenizer.texts_to_sequences(positive_text)
negative_sequences = tokenizer.texts_to_sequences(negative_text)

max_sequence_length = max(max(map(len, anchor_sequences)), max(map(len, positive_sequences)), max(map(len, negative_sequences)))

anchor_padded = pad_sequences(anchor_sequences, maxlen=max_sequence_length, padding='post')
positive_padded = pad_sequences(positive_sequences, maxlen=max_sequence_length, padding='post')
negative_padded = pad_sequences(negative_sequences, maxlen=max_sequence_length, padding='post')


min_count = min(len(anchor_padded), len(positive_padded), len(negative_padded))
anchor_padded = anchor_padded[:min_count]
positive_padded = positive_padded[:min_count]
negative_padded = negative_padded[:min_count]


split_ratio = [0.8, 0.1, 0.1]
total_samples = len(anchor_padded)
split_index_1 = int(total_samples * split_ratio[0])
split_index_2 = int(total_samples * (split_ratio[0] + split_ratio[1]))

train_anchor = anchor_padded[:split_index_1]
train_positive = positive_padded[:split_index_1]
train_negative = negative_padded[:split_index_1]

val_anchor = anchor_padded[split_index_1:split_index_2]
val_positive = positive_padded[split_index_1:split_index_2]
val_negative = negative_padded[split_index_1:split_index_2]

test_anchor = anchor_padded[split_index_2:]
test_positive = positive_padded[split_index_2:]
test_negative = negative_padded[split_index_2:]


siamese_model = create_siamese_model(vocab_size, embedding_dim, lstm_units)

siamese_model.compile(loss=triplet_loss, optimizer=Adam(learning_rate=0.001))

siamese_model.fit(
    [train_anchor, train_positive, train_negative],
    [np.ones_like(train_anchor), np.ones_like(train_anchor)],
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(
        [val_anchor, val_positive, val_negative],
        [np.ones_like(val_anchor), np.ones_like(val_anchor)]
    )
)

loss = siamese_model.evaluate(
    [test_anchor, test_positive, test_negative],
    [np.ones_like(test_anchor), np.ones_like(test_anchor)]
)

print("Test loss:", loss)


