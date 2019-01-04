"""
At the lowest level we have "words".  These are the objects
that we want to produce vectors for.  They are grouped into
"sentences" but they need not be sentences.  They are just
groups of words we will scan windows over.

https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py
https://github.com/cbellei/word2veclite


skipgram:
  predict context word(s) from target word

cbow:
  predict target word from context word(s)

"""
import string

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from space_docs_big_idea import SpaceDocsBigIdea

TOL = 1.0e-7

corpus = SpaceDocsBigIdea('space_docs_big_idea.txt')

# create vocab
word2index = {}
index2word = {}
for sentence in corpus.iter_sentences():
    for word in sentence:
        if word not in word2index:
            word2index[word] = len(word2index)
index2word = {index:word for word, index in word2index.items()}


back_window = 2
front_window = 2
embd_size = 20
vocab_size = len(word2index)
learning_rate = 0.05
n_epochs = 100
batch_size = 500
shuffle = True


# create weight matrices
fac = np.sqrt(6) / np.sqrt(embd_size + vocab_size)
W1 = fac * (np.random.rand(vocab_size, embd_size) - 0.5)
W2 = fac * (np.random.rand(embd_size, vocab_size) - 0.5)

# one update per batch (mini batch gradient descent)
J_history = []
isample = 0
ibatch = 0


def cbow_on_sample(sample, vocab_size, word2index):

    target_word, context_words = sample

    # create one-hot encoded target word vector
    target_index = word2index[target_word]
    target_1hot = np.zeros((vocab_size, 1))
    target_1hot[target_index, 0] = 1

    # create one-hot encoded context word vectors
    context_indices = [word2index[word] for word in context_words]
    context_1hots = []
    for context_index in context_indices:
        context_1hot = np.zeros((vocab_size, 1))
        context_1hot[context_index, 0] = 1
        context_1hots.append(context_1hot)

    # create average one-hot encoded context vector (i.e. the input)
    x_1hot = np.zeros((vocab_size, 1))
    for context_1hot in context_1hots:
        x_1hot += context_1hot / len(context_indices)

    # calculate hidden activations with matmul
    h_mm = np.matmul(W1.T, x_1hot)

    # calculate hidden activations by selecting/averaging rows
    h_se = W1[context_indices, :].T.sum(axis=1, keepdims=True) / len(context_indices)

    assert(np.abs(np.sum(h_mm - h_se)) < TOL)
    hh = h_se

    # create one-hot encoded target word vector (i.e. the output)
    y_1hot = np.zeros((vocab_size, 1))
    y_1hot[target_index, 0] = 1

    # calculate output vector
    uu = np.matmul(W2.T, hh)
    expu = np.exp(uu)
    yy = expu / np.sum(expu)

    # calculate sample loss ... Jsamp = -log p(w_target|w_context)
    Jsamp = -(uu[target_index, 0] - np.log(np.sum(expu)))

    # calculate output error
    dedu = yy - y_1hot

    # calculate update to W2
    dedw2 = np.matmul(hh, dedu.T)

    # calculate update to W1
    dedh = np.matmul(W2, dedu)
    dedw1 = np.matmul(x_1hot, dedh.T)

    return Jsamp, dedw1, dedw2


for iepoch in range(n_epochs):
    print('iepoch = {}'.format(iepoch))

    corpus_iter = corpus.iter_target_context_batch(
        back_window=back_window,
        front_window=front_window,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    for batch in corpus_iter:
        ibatch += 1

        dedw1_batch = np.zeros_like(W1)
        dedw2_batch = np.zeros_like(W2)
        J_batch = 0
        for sample in batch:

            target_word, context_words = sample
            if len(context_words) == 0:
                continue
            isample += 1

            J_sample, dedw1_sample, dedw2_sample = cbow_on_sample(sample, vocab_size, word2index)
            dedw1_batch += dedw1_sample
            dedw2_batch += dedw2_sample
            J_batch += J_sample

        W1 -= learning_rate * dedw1_batch
        W2 -= learning_rate * dedw2_batch
        J_batch /= len(batch)
        J_history.append(J_batch)
        print('ibatch={}, J_batch={}'.format(ibatch, J_batch))


pca = PCA(n_components=2)
for index, word in index2word.items():
    print(index, word)
    sims = W1.dot(W1[index,:])
    max_indices = np.argsort(-sims)[:10]
    for ii in max_indices:
        print('    ', index2word[ii])
