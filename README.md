# Week 8: Encoder-Decoder Architecture for Sequence-to-Sequence Tasks
Encoder-Decoder architecture is a type of model used in many sequence-to-sequence tasks in Natural Language Processing (NLP). These tasks, as the name suggests, involve converting an input sequence into an output sequence, and include machine translation, text summarization, and question answering among others.

How does the Encoder-Decoder Architecture Work?
The Encoder-Decoder architecture is made up of two main components: the encoder and the decoder, both of which are often implemented as Recurrent Neural Networks (RNNs) or more advanced variants like LSTMs (Long Short-Term Memory) or GRUs (Gated Recurrent Units).

Encoder: The encoder processes the input sequence and summarizes the information in its internal state, often referred to as the context or thought vector. This vector is expected to be a good representation of the entire input sequence.

Decoder: The decoder generates the output sequence step-by-step based on the context vector provided by the encoder and what it has generated so far. In other words, the decoder is an RNN that generates probabilities for the output sequence. For each step, the decoder is influenced by the context vector and the previously generated words.

While the simplest versions of the encoder and decoder share the same weights, more complex versions may use different parameters for the two components. The architecture is flexible and allows for many variations, such as the use of multiple layers of encoders and decoders.

Applications of the Encoder-Decoder Architecture
A classic example of the use of an Encoder-Decoder architecture is in machine translation, where the input sequence is a sentence in the source language, and the output sequence is the corresponding translation in the target language. The encoder transforms the source sentence into a context vector, and then the decoder generates the target sentence based on this vector.

Another example is text summarization, where the input is a long document, and the output is a shorter summary. The encoder reads the entire document and encodes it into a context vector. The decoder then uses this vector to generate a brief summary.

Challenges and Limitations
While Encoder-Decoder architectures have proven successful for many tasks, they are not without their limitations. One key issue is that the encoder must compress all the necessary information of the input sequence into a fixed-size context vector, which can lead to information loss, especially for longer sequences.

To overcome this limitation, attention mechanisms have been introduced, which allow the model to focus on different parts of the input sequence at each step of the output generation, instead of encoding the entire sequence into a single context vector. This has been a significant improvement and is now a standard component of many sequence-to-sequence models.

# Readings

[Sentence-Level Grammatical Error Identification as Sequence-to-Sequence Correction](https://arxiv.org/pdf/1604.04677.pdf)


[Understanding How Encoder-Decoder Architectures Attend](https://proceedings.neurips.cc/paper_files/paper/2021/file/ba3c736667394d5082f86f28aef38107-Paper.pdf)


[Evaluating Sequence-to-Sequence Models for Handwritten Text Recognition](https://arxiv.org/pdf/1903.07377.pdf)
