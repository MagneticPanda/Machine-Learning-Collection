# About
Seq2seq models have seen fruitful use across various tasks such as text summarization, speech recognition, question answering, and much more [1]. They are preferred over other sequence modelling architectures due to their nature of being able to handle variable length input and/or output sequences (i.e., when the input or output is not of a fixed size). In this particular report, we will be looking at applying these 3 architectures in a machine translation task. More specifically, we shall be performing a character-level machine translation task of converting short English expressions to French.

This project compares and contrasts 3 different sequence-to-sequence (seq2seq) architectures:

1. Bidirectional LSTM encoder and an LSTM decoder
2. Bidirectional GRU encoder and an GRU decoder
3. 2-Layered LSTM encoder and an LSTM decoder
