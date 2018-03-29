"""
At the lowest level we have "words".  These are the objects
that we want to produce vectors for.  They are grouped into
"sentences" but they need not be sentences.  They are just
groups of words we will scan windows over.

https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py
https://github.com/cbellei/word2veclite


skipgram:
  predict context word(s) from center word

cbow:
  predict center word from context word(s)

"""
import matplotlib.pyplot as plt
import numpy as np
TOL = 1.0e-7

corpus = ['I like playing football with my friends'.split()]

# create vocab
word2indx = {}
indx2word = {}
for sentence in corpus:
    for word in sentence:
        if word not in word2indx:
            word2indx[word] = len(word2indx)


back_window = 2
front_window = 2


embd_size = 5
vocab_size = len(word2indx)
learning_rate = 0.01
n_epochs = 300

# create weight matrices
W1 = 1.0 * (np.random.rand(vocab_size, embd_size) - 0.5)
W2 = 1.0 * (np.random.rand(embd_size, vocab_size) - 0.5)


def create_batch(corpus, back_window, front_window, word2indx):
    word_tups = []
    indx_tups = []
    for sentence in corpus:
        for si_center_word, center_word in enumerate(sentence):
            si_min_context_word = max(0, si_center_word - back_window)
            si_max_context_word = min(len(sentence)-1, si_center_word + front_window)
            si_context_words = [
                si for si in range(si_min_context_word, si_max_context_word + 1)
                if si != si_center_word]
            context_words = [sentence[si] for si in si_context_words]
            center_indx = word2indx[center_word]
            context_indxs = [word2indx[word] for word in context_words]

            print(center_word, context_words)
            word_tups.append((center_word, context_words))
            indx_tups.append((center_indx, context_indxs))
    return indx_tups




# one update per epoch (batch SGD)
# CBOW: input = average context words
#       output = center word
Jarr = []
for iepoch in range(n_epochs):
    print('iepoch = {}'.format(iepoch))
    dedw1 = np.zeros_like(W1)
    dedw2 = np.zeros_like(W2)
    Jtot = 0
    isample = 0
    indx_tups = create_batch(corpus, back_window, front_window, word2indx)

    for center_indx, context_indxs in indx_tups:
        isample += 1

        # create one-hot encoded center word vector (i.e. the output)
        y_1hot = np.zeros((vocab_size, 1))
        y_1hot[center_indx, 0] = 1

        # create average one-hot encoded context vector (i.e. the input)
        x_1hot = np.zeros((vocab_size, 1))
        x_1hot[context_indxs, 0] = 1/len(context_indxs)

        # calculate hidden activations with matmul
        h_mm = np.matmul(W1.T, x_1hot)

        # calculate hidden activations by selecting/averaging rows
        h_se = W1[context_indxs, :].T.sum(axis=1, keepdims=True) / len(context_indxs)

        assert(np.abs(np.sum(h_mm - h_se)) < TOL)
        h = h_se

        # calculate output vector
        u = np.matmul(W2.T, h)
        expu = np.exp(u)
        y = expu / np.sum(expu)

        # calculate training sample loss ... Jsamp = -log p(w_context|w_center)
        Jsamp = -(u[center_indx, 0] - np.log(np.sum(expu)))
        Jtot += Jsamp

        # calculate output error
        dedu = y - y_1hot

        # calculate update to W2
        dedw2 += np.matmul(h, dedu.T)

        # calculate update to W1
        dedh = np.matmul(W2, dedu)
        dedw1 += np.matmul(x_1hot, dedh.T)

    W1 -= learning_rate * dedw1
    W2 -= learning_rate * dedw2
    Jtot /= isample
    Jarr.append(Jtot)
    print('Jtot={}'.format(Jtot))
