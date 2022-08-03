"""
NAME: Sashen Moodley
Student Number: 219006946
"""

import warnings
import time
import keras.utils.np_utils
import numpy as np
import keras.utils.np_utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN, KMeans
from imblearn.over_sampling import RandomOverSampler
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras import backend as K
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from matplotlib import rcParams
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings('ignore')
rcParams.update({'figure.autolayout': True})  # To ensure proper scaling of visuals

# GLOBAL VARIABLES
NO_OF_PRICES = 8
EPOCHS = 100
BATCH_SIZE = 128


# -------------------Legacy Metric Functions-------------------------
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


# -----------------------------------------------------------------

# --------------Standard Normal Distribution Scaler ---------------
def scale_data(df):
    df_transpose = df.transpose()

    scaler = StandardScaler()
    df_transpose = scaler.fit_transform(df_transpose)

    df = pd.DataFrame(df_transpose.transpose())

    return df

# -----------------------------------------------------------------

# --------------Plotting/Display  Functions------------------------
def plot_graphs(model_type, model_history, metric, plot_title):
    plt.plot(model_history.history[metric], label=f'Train {metric}')
    plt.plot(model_history.history['val_' + metric], label=f'Validation {metric}')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend()
    plt.title(plot_title)
    if model_type == 'LSTM':
        save_to = f'LSTM Plots/{plot_title}.png'
    else:
        save_to = f'GRU Plots/{plot_title}.png'
    plt.savefig(save_to)
    plt.show()


def display_scores(scores, model_type: str):
    print(f"---------{model_type} EVALUATION STATS---------")
    print(f"Loss: {scores[0]}")
    print(f"Accuracy: {scores[1]}")
    print(f"Mean Squared Error: {scores[2]}")
    print(f"F1-Score: {scores[3]}")
    print(f"Precision: {scores[4]}")
    print(f"Recall: {scores[5]}")


def plot_time_series(df, plot_title, type=None):
    df.transpose().plot(legend=False)
    plt.title(plot_title)
    if type == 'KCluster':
        save_to = f'KMeans Clusters/{plot_title}.png'
    elif type == 'DBCluster':
        save_to = f'DBSCAN Clusters/{plot_title}.png'
    else:
        save_to = f'Trend Visualization/{plot_title}.png'
    plt.savefig(save_to)
    plt.show()


# -----------------------------------------------------------------

# ==================== MAIN APPLICATION ===========================

# Loading the data we got from the web scraping application
turnip_dataframe = pd.read_csv('TurnipsDS.csv')
turnip_dataframe = turnip_dataframe[
    ['Mon_AM', 'Mon_PM', 'Tues_AM', 'Tues_PM', 'Wed_AM', 'Wed_PM',
     'Thurs_AM', 'Thurs_PM', 'Fri_AM', 'Fri_PM', 'Sat_AM', 'Sat_PM']]
turnip_dataframe = turnip_dataframe.astype(float)
print(turnip_dataframe)
print(turnip_dataframe.info())

# ------------- Exploratory Data Analysis ------------------

# Visualizing the data to see if we can identify community-believed price trends
plot_time_series(turnip_dataframe.head(10), plot_title='Trend Visualization', type='Trend Visualization')

# 2-D Data projections, for visual detection of pairwise correlations to aid in clustering
sns.pairplot(turnip_dataframe)
plt.savefig('Miscellaneous Plots/Pairwise Correlation.png')
plt.show()

# Some kernel density plots for daily price distribution
for col in turnip_dataframe.columns:
    sns.kdeplot(turnip_dataframe[col])
    plt.xlim(0, 700)
    plt.title(col)
    plt.savefig(f'Daily Kernel Density Plots/{str(col)} density plot.png')
    plt.show()


# -----------------------------------------------------------

# -------------------Preprocessing---------------------------

# The data is scaled to a standard normal distribution rather than in the interval [0,1] as this would eliminate
# the spikes in the data which is vital for our trend identification
scaled_data = scale_data(turnip_dataframe)
plot_time_series(scaled_data.head(5), plot_title='Trend Visualization after Scaling', type='Trend Visualization')

# ----------------------------------------------------------

# ---------------- Clustering Analysis ---------------------

# K-MEANS CLUSTERING
clusters = KMeans(n_clusters=12, init='k-means++').fit_predict(scaled_data)
turnip_dataframe['kmeans'] = clusters

# # plotting the samples belonging to each cluster
for i in range(12):
    plot_time_series(turnip_dataframe.loc[turnip_dataframe['kmeans'] == i].iloc[0:250, :-1],
                     plot_title=f'K-Means Cluster {i}', type='KCluster')

turnip_dataframe.drop('kmeans', axis=1, inplace=True)

# DBSCAN CLUSTERING
dbscan = DBSCAN(eps=0.5, min_samples=18).fit_predict(scaled_data)
turnip_dataframe['dbscan'] = dbscan
print(turnip_dataframe)

# plotting the samples belonging to each cluster
for i in range(-1, dbscan.max() + 1):
    plot_time_series(turnip_dataframe.loc[turnip_dataframe['dbscan'] == i].iloc[0:250, :-2],
                     plot_title=f'DBSCAN Cluster {i}', type='DBCluster')

# Showing the DBSCAN label distribution
sns.countplot(turnip_dataframe['dbscan'])
plt.savefig('Miscellaneous Plots/Unevenly distributed countplot.png')
plt.show()

# ---------------------------------------------------------

# ----------------- Preparing the Data --------------------

# Getting information about DBSCAN labels for scaling and encoding later on
class_counts = turnip_dataframe['dbscan'].value_counts()
cols = turnip_dataframe.columns[turnip_dataframe.columns != 'dbscan']
x_ = turnip_dataframe.loc[:, cols]
y_ = turnip_dataframe['dbscan']
print(f"Total number of samples: {len(x_)}")

# Performing random over-sampling to get even distrubution of samples across the DBSCAN labels
ros = RandomOverSampler(sampling_strategy='auto', random_state=27)
x_res, y_res = ros.fit_resample(x_, y_)
print(f"Total number of samples after Random Over Sampling: {len(x_res)}")

# Visualization of random-over sampling on the distribution
sns.countplot(y_res)
plt.savefig('Miscellaneous Plots/ROS distributed countplot.png')
plt.show()

# Splitting the data 80:20
x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, train_size=0.8, random_state=24)
print(f"There are {len(x_train)} training samples")
print(f"There are {len(x_test)} testing samples")

# Scaling normally distributed samples
scaler = MinMaxScaler()
scaler.fit(x_train.iloc[:, :NO_OF_PRICES])
x_train = scaler.transform(x_train.iloc[:, :NO_OF_PRICES])
x_test = scaler.transform(x_test.iloc[:, :NO_OF_PRICES])

# Visualization of sample distribution over labels after split is made
sns.countplot(y_train)
plt.savefig('Miscellaneous Plots/After split distributed countplot.png')
plt.show()

# Encoding labels
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.fit_transform(y_test)
y_train_cat = keras.utils.np_utils.to_categorical(y_train_enc, num_classes=len(class_counts), dtype='float32')
y_test_cat = keras.utils.np_utils.to_categorical(y_test_enc, num_classes=len(class_counts), dtype='float32')

# Required reshaping when creating predictions
y_test_cat_pred = np.argmax(y_test_cat, axis=1)

# ---------------------------------------------------------

# ---------------NAIVE BAYES CLASSIFICATION----------------

naive_classifier = GaussianNB()

# Fitting data to naive classifier
start_time = time.time()
naive_model = naive_classifier.fit(x_train[:, :NO_OF_PRICES], y_train_enc)
end_time = time.time()
print(f"Naive Classifier took {end_time - start_time}s to fit data")

# Creating predictions on naive classifier
start_time = time.time()
naive_predictions = naive_model.predict(x_test[:, :NO_OF_PRICES])
end_time = time.time()
print(f"Naive  Classifier took {end_time - start_time}s to make predictions")

# Creating confusion matrix
confuse_matrix_naive = confusion_matrix(y_true=y_test_enc, y_pred=naive_predictions)
disp = ConfusionMatrixDisplay(confuse_matrix_naive)
disp.plot()
plt.title("Naive Classifier Confusion Matrix")
plt.savefig('Confusion Matrices/Naive Bayes confusion matrix.png')
plt.show()

# Getting the classification report from predictions
print("------NAIVE BAYES CLASSIFICATION REPORT---------")
print(classification_report(y_true=y_test_enc, y_pred=naive_predictions))

# ---------------------------------------------------------

# ------------------ LSTM Classifier ----------------------

# Reshaping data for input layer of both RNNs
x_train = x_train.reshape(-1, NO_OF_PRICES, 1)
x_test = x_test.reshape(-1, NO_OF_PRICES, 1)

# Setting up the model architecture
turnip_classifier_lstm = Sequential([
    Dense(64, activation='relu', input_shape=(NO_OF_PRICES, 1), name='Input_Layer'),
    LSTM(64, return_sequences=False, name='LSTM_Layer'),
    Dense(len(class_counts), activation='softmax', name='Softmax_Output_Layer')
], name='LSTM-Turnip-Price-Classifier')
turnip_classifier_lstm.summary()

# Compiling with appropriate metrics
turnip_classifier_lstm.compile(optimizer='adam',
                               loss='categorical_crossentropy',
                               metrics=['accuracy', 'mean_squared_error', f1_m, precision_m, recall_m])

# Setting up callbacks
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=10, verbose=2, factor=0.5, min_lr=0.00001)
lstm_best_model = ModelCheckpoint('lstm_best_model.h5', monitor='val_accuracy', verbose=2, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=25, restore_best_weights=True)

# Training the LSTM
start_time = time.time()
model_lstm = turnip_classifier_lstm.fit(x=x_train, y=y_train_cat,
                                        batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1,
                                        callbacks=[learning_rate_reduction, lstm_best_model, early_stopping])
end_time = time.time()
print(f"LSTM Model took {end_time - start_time}s to train on data")

# Plotting the metrics gathered through training and validation
plot_graphs('LSTM', model_lstm, 'loss', 'LSTM Loss')
plot_graphs('LSTM', model_lstm, 'accuracy', 'LSTM Accuracy')
plot_graphs('LSTM', model_lstm, 'mean_squared_error', 'LSTM MSE')
plot_graphs('LSTM', model_lstm, 'f1_m', 'LSTM F1 Score')
plot_graphs('LSTM', model_lstm, 'precision_m', 'LSTM Precision')
plot_graphs('LSTM', model_lstm, 'recall_m', 'LSTM Recall')

# Performing evaluation to gather metric scores
start_time = time.time()
scores_lstm = turnip_classifier_lstm.evaluate(x=x_test, y=y_test_cat)
end_time = time.time()
print(f"LSTM took {end_time - start_time}s to perform evaluations")
display_scores(scores_lstm, 'LSTM')

# Performing predictions on testing data
start_time = time.time()
predictions_lstm = turnip_classifier_lstm.predict(x=x_test, batch_size=BATCH_SIZE)
end_time = time.time()
print(f"LSTM took {end_time - start_time}s to make predictions")
predictions_lstm = np.argmax(predictions_lstm, axis=1)

# Getting classification report based on predictions
print("--------LSTM CLASSIFICATION REPORT---------")
print(classification_report(y_true=y_test_cat_pred, y_pred=predictions_lstm))

# Plotting confusion matrix based on predictions
confuse_matrix_lstm = confusion_matrix(y_true=y_test_cat_pred, y_pred=predictions_lstm)
disp = ConfusionMatrixDisplay(confuse_matrix_lstm)
disp.plot()
plt.title("LSTM Confusion Matrix")
plt.savefig('Confusion Matrices/LSTM confusion matrix.png')
plt.show()

# ---------------------------------------------------------

# ------------------ GRU Classifier -----------------------

# # Setting up the model architecture
turnip_classifier_gru = Sequential([
    Dense(64, activation='relu', input_shape=(NO_OF_PRICES, 1), name='Input_Layer'),
    GRU(64, return_sequences=False, name='GRU_Layer'),
    Dense(len(class_counts), activation='softmax', name='Softmax_Output_Layer')
], name='GRU-Turnip-price-Classifier')
turnip_classifier_gru.summary()

# Compiling with appropriate metrics
turnip_classifier_gru.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy', 'mean_squared_error', f1_m, precision_m, recall_m])

# Setting up callbacks
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=10, verbose=2, factor=0.5, min_lr=0.00001)
gru_best_model = ModelCheckpoint('gru_best_model.h5', monitor='val_accuracy', verbose=2, save_best_only=True,mode='max')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=25, restore_best_weights=True)

# Training the GRU
start_time = time.time()
model_gru = turnip_classifier_gru.fit(x=x_train, y=y_train_cat,
                                      batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1,
                                      callbacks=[learning_rate_reduction, gru_best_model, early_stopping])
end_time = time.time()
print(f"GRU Model took {end_time - start_time}s to train on data")

# Plotting the metrics gathered through training and validation
plot_graphs('GRU', model_gru, 'loss', 'GRU Loss')
plot_graphs('GRU', model_gru, 'accuracy', 'GRU Accuracy')
plot_graphs('GRU', model_gru, 'mean_squared_error', 'GRU MSE')
plot_graphs('GRU', model_gru, 'f1_m', 'GRU F1 Score')
plot_graphs('GRU', model_gru, 'precision_m', 'GRU Precision')
plot_graphs('GRU', model_gru, 'recall_m', 'GRU Recall')

# Performing evaluation to gather metric scores
start_time = time.time()
scores_gru = turnip_classifier_gru.evaluate(x=x_test, y=y_test_cat)
end_time = time.time()
print(f"GRU took {end_time - start_time}s to perform evaluations")
display_scores(scores_gru, 'GRU')

# Performing predictions on testing data
start_time = time.time()
predictions_gru = turnip_classifier_gru.predict(x=x_test, batch_size=BATCH_SIZE)
end_time = time.time()
print(f"GRU took {end_time - start_time}s to make predictions")
predictions_gru = np.argmax(predictions_gru, axis=1)

# Getting classification report based on predictions
print("-------GRU CLASSIFICATION REPORT---------")
print(classification_report(y_true=y_test_cat_pred, y_pred=predictions_gru))

# Plotting confusion matrix based on predictions
confuse_matrix_gru = confusion_matrix(y_true=y_test_cat_pred, y_pred=predictions_gru)
disp = ConfusionMatrixDisplay(confuse_matrix_gru)
disp.plot()
plt.title("GRU Confusion Matrix")
plt.savefig('Confusion Matrices/GRU confusion matrix.png')
plt.show()

# ===============================================================
