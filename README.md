# Machine-Learning-Collection
### About
Name: Sashen Moodley

This repository contains an ever growing list of machine learning projects that I have created.

# Projects
A short description of each project is provided below. For more information please refer to the _report_ within the respective project folder.

## Stock Market Classifier and Predictor
Sequence modelling has seen fruitful use in various applications such as speech recognition, music generation, sentiment classification, machine translation, and time series prediction. However, an area that is not widely studied is the upcoming time series classification variant seen in artificial game markets. With various potential candidate artificial game markets to analyse, I was interested in implementing a time series classification for a personal favourite - Nintendo’s Animal Crossing: New Horizons (ACNH).

![ACNH](https://user-images.githubusercontent.com/71750671/182713835-4de2805c-906b-4e0f-8102-54f146857693.jpg)

In this project I implemented two RNN-based architectures (LSTM and GRU) and compared them against the more traditional naïve classifiers. The main task of this project is to classify the various weekly price trends that a player's island might follow - decreasing, high spike, low spike, and random. However, through DBSCAN clustering a more expressive classification could be created by carrying time-related semantics within the classification itself, hence enabling the model to create predictions via it's predicted price trend classification.
This project demonstated that sequence modelling techniques are valid, and incredibly useful, in artifical game markets as it outperforms statistically naïve models by an average of 6%, and this margin only increases in favour of sequence models when longer sequences are observed.

## South African Bank Notes Recognition System
Image processing and computer vision - Created a robust bank notes recognition system evaluated various image pre-processing, segmentation, feature extraction and classification techniques techniques to build a robust system which achieves a 93% accuracy during testing, and is capable of being deployed in ATMs, vending machines as well as aid visually impaired  individuals.

![tinywow_resize_4083655](https://user-images.githubusercontent.com/71750671/182954544-e8169382-0bbd-4139-baad-c7bb8057f4be.jpg)

## isiZulu Part of Speech Tagging
Implemented a Conditional Random Field (CRF) and Hidden Markov Model (HMM) to perform part of speech tagging on a large corpus of isiZulu sentences. The efficacy of these two models were empirically evaluated, which ultimately concluded with the view that the CRF outperforms the HMM for this task.

![eight-parts-speech_0066f46bde](https://user-images.githubusercontent.com/71750671/182952180-456a6df4-8389-4e94-862f-7822bc83d738.jpg)

## Twitter Sentiment Classification
Implemented LSTM and GRU models for the task of classifying a corpus of tweets based on their underlying sentiment. These two models were emperically evaluated, which found that the GRU was the preferred model.

![twitter](https://user-images.githubusercontent.com/71750671/182952749-b303fda5-6e2a-46da-b49d-5c798a46eacb.jpg)

## English to French Translator
Implemented a character level translator utilizing 3 differet sequence-to-sequence (seq2seq) model architectures:

1. Bidirectional LSTM encoder and an LSTM decoder
2. Bidirectional GRU encoder and an GRU decoder
3. 2-Layered LSTM encoder and an LSTM decoder

These 3 models were empirically evaluated which ultimately found that the GRU is faster to train and make predictions, but the LSTM provides slightly more accurate predictions (when using the same bidirectional philosophy). Furthermore it was shown that bidirectional architectures provide more accurate translations than
a stacked architecture in this task.

![tinywow_resize_4083638](https://user-images.githubusercontent.com/71750671/182954575-06ee441b-ced2-4c71-99b1-af581dce53b0.jpg)
