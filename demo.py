"""
At the lowest level we have "words".  These are the objects
that we want to produce vectors for.  They are grouped into
"sentences" but they need not be sentences.  They are just
groups of words we will scan windows over.

https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py
https://github.com/cbellei/word2veclite

"""
import matplotlib.pyplot as plt
import numpy as np
TOL = 1.0e-7

corpus = ['I like playing football with my frients'.split()]

# create vocab
word2indx = {}
indx2word = {}
for sentence in corpus:
    for word in sentence:
        if word not in word2indx:
            word2indx[word] = len(word2indx)


back_window = 1
front_window = 1


embd_size = 5
vocab_size = len(word2indx)
learning_rate = 0.01
n_epochs = 300

# create weight matrices
W1 = 1.0 * (np.random.rand(vocab_size, embd_size) - 0.5)
W2 = 1.0 * (np.random.rand(embd_size, vocab_size) - 0.5)


# one update per epoch (batch SGD)
Jarr = []
for iepoch in range(n_epochs):
    print('iepoch = {}'.format(iepoch))
    dedw1 = np.zeros_like(W1)
    dedw2 = np.zeros_like(W2)
    Jtot = 0
    isample = 0

    for sentence in corpus:
        for i_center_word, center_word in enumerate(sentence):
            i_min_context_word = max(0, i_center_word - back_window)
            i_max_context_word = min(len(sentence), i_center_word + front_window + 1)
            for i_context_word in range(i_min_context_word, i_max_context_word):
                if i_center_word == i_context_word:
                    continue

                isample += 1
                context_word = sentence[i_context_word]
                center_indx = word2indx[center_word]
                context_indx = word2indx[context_word]
                #print(center_word, center_indx, context_word, context_indx)

                # create one-hot encoded center word vector
                x_1hot = np.zeros((1, vocab_size))
                x_1hot[0, center_indx] = 1

                # create one-hot encoded context word vector
                y_1hot = np.zeros((1, vocab_size))
                y_1hot[0, context_indx] = 1

                # calculate hidden activations with matmul
                h_mm = np.matmul(x_1hot, W1)

                # calculate hidden activations by selection a row
                h_se = W1[center_indx:center_indx+1, :]

                assert(np.abs(np.sum(h_mm - h_se)) < TOL)
                h = h_se

                # calculate output vector
                u = np.matmul(h, W2)
                expu = np.exp(u)
                y = expu / np.sum(expu)

                # calculate training sample loss ... Jsamp = -log p(w_context|w_center)
                Jsamp = -(u[0, context_indx] - np.log(np.sum(expu)))
                Jtot += Jsamp
                #print('Jsamp={}'.format(Jsamp))

                # calculate output error
                dedu = y - y_1hot

                # calculate update to W2
                dedw2 += np.matmul(h.T, dedu)

                # calculate update to W1
                dedh = np.matmul(dedu, W2.T)
                dedw1 += np.matmul(x_1hot.T, dedh)

    W1 -= learning_rate * dedw1
    W2 -= learning_rate * dedw2
    Jtot /= isample
    Jarr.append(Jtot)
    print('Jtot={}'.format(Jtot))
