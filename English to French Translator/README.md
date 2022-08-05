# About
Seq2seq models have seen fruitful use across various tasks such as text summarization, speech recognition, question answering, and much more. They are preferred over other sequence modelling architectures due to their nature of being able to handle variable length input and/or output sequences (i.e., when the input or output is not of a fixed size). In this particular report, we will be looking at applying these 3 architectures in a machine translation task. More specifically, we shall be performing a character-level machine translation task of converting short English expressions to French.

This project compares and contrasts 3 different sequence-to-sequence (seq2seq) architectures:

1. Bidirectional LSTM encoder and an LSTM decoder
2. Bidirectional GRU encoder and an GRU decoder
3. 2-Layered LSTM encoder and an LSTM decoder

# Summary of Results and Conclusion


## Dataset Acquisition
Note: A custom dataset, referenced as 'fra.txt' in the source files, was used as the labelled corpus. Due to size constaints this has been omitted from the repository, however a copy of this dataset can be provided upon request from [this google drive link](https://drive.google.com/file/d/1kTqw-yNw4UnXOaJ0JBUcFE1Ez2bALrdD/view?usp=sharing)
