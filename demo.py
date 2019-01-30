"""
Pedagogical implementation of word2vec.

References
https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py
https://github.com/cbellei/word2veclite

cbow:
  predict target word from context word(s)

skipgram:
  predict context word(s) from target word


"""
from collections import Counter
import json
import string
import numpy as np
import os
TOL = 1.0e-7

# corpora are pre-tokenized in json format
# each corpus is a list of lists of words
# vocabs are lists of word, count pairs sorted by frequency
#-------------------------------------------

corpus_name = 'space_docs_big_idea'
#corpus_name = 'tng_scripts'

corpus_path = os.path.join('corpora', corpus_name)
corpus_file = os.path.join(corpus_path, 'corpus.json')
vocab_file = os.path.join(corpus_path, 'vocab.json')

with open(corpus_file, 'r') as fp:
    corpus = json.load(fp)
with open(vocab_file, 'r') as fp:
    vocab = json.load(fp)


# create word to index mapping and inverse
#-------------------------------------------
word2index = {}
index2word = {}
for index, (word, count) in enumerate(vocab):
    word2index[word] = len(word2index)
index2word = {index:word for word, index in word2index.items()}


# set word2vec parameters
#-------------------------------------------
back_window = 4
front_window = 4
learning_rate = 0.05
shuffle = True
vocab_size = len(vocab)
if corpus_name == 'space_docs_big_idea':
    embd_size = 10
    batch_size = 50
    n_epochs = 500
elif corpus_name == 'tng_scripts':
    embd_size = 100
    batch_size = 5000
    n_epochs = 2
else:
    raise ValueError('corpus name not recognized')



def cbow_on_sample(sample, vocab_size, word2index):
    """Continuous Bag of Words model on one sample."""

    target_word, context_words = sample

    # create (V x 1) average "one-hot" encoded context word vectors for input
    context_indices = [word2index[word] for word in context_words]
    cfac = 1 / len(context_indices)
    x_1hot = np.zeros((vocab_size, 1))
    for context_index in context_indices:
        x_1hot[context_index, 0] += cfac

    # create (V x 1) one-hot encoded target word vector for output
    target_index = word2index[target_word]
    y_1hot = np.zeros((vocab_size, 1))
    y_1hot[target_index, 0] = 1

    # calculate hidden activations with matmul
    # (N x V) . (V x 1) = (N x 1)
    h_mm = np.matmul(W1.T, x_1hot)

    # calculate hidden activations by selecting/averaging rows
    # W1 has shape (V x N) so W1.T has shape (N x V)
    # selecting rows from W1 and then transposing give h_se.shape = (N,1)
    h_se = W1[context_indices, :].T.sum(axis=1, keepdims=True) * cfac

    assert(np.max(np.abs(h_mm - h_se)) < TOL)
    hh = h_se

    # calculate output vector
    # W2 has shape (N x V) so W2.T has shape (V x N)
    # uu has shape (V x N) . (N x 1) = (V x 1)
    uu = np.matmul(W2.T, hh)

    # create a probability distribution (yy) via softmax
    expu = np.exp(uu)
    prob_norm = np.sum(expu)
    yy = expu / prob_norm

    # calculate sample loss ... Jsamp = -log p(w_target|w_context)
    Jsamp = -(uu[target_index, 0] - np.log(prob_norm))

    # calculate output error
    dedu = yy - y_1hot

    # calculate update to W2
    dedw2 = np.matmul(hh, dedu.T)

    # calculate update to W1
    dedh = np.matmul(W2, dedu)
    dedw1 = np.matmul(x_1hot, dedh.T)

    return Jsamp, dedw1, dedw2



def target_context_samples_from_sentence(sentence):
    # iterate over target words in sentence
    samples = []
    for target_indx, target_word in enumerate(sentence):
        min_context_indx = max(0, target_indx - back_window)
        max_context_indx = min(len(sentence)-1, target_indx + front_window)
        context_indices = [
            indx for indx in range(min_context_indx, max_context_indx + 1)
            if indx != target_indx]
        context_words = [sentence[indx] for indx in context_indices]
        samples.append((target_word, context_words))
    return samples


def grouper(iterable, n):
    """Collect data into fixed-length chunks or blocks
    https://docs.python.org/3/library/itertools.html"""
    args = [iter(iterable)] * n
    return zip(*args)



all_samples = []
# iterate over sentences in corpus
for sentence in corpus:
    all_samples.extend(target_context_samples_from_sentence(sentence))
print('created {len(all_samples)} samples')


# create/read weight matrices
#-------------------------------------------

#fac = np.sqrt(6) / np.sqrt(embd_size + vocab_size)
#W1 = fac * (np.random.rand(vocab_size, embd_size) - 0.5)
#W2 = fac * (np.random.rand(embd_size, vocab_size) - 0.5)
#np.savetxt(f'W1_{corpus_name}.csv', W1, delimiter=',')
#np.savetxt(f'W2_{corpus_name}.csv', W2, delimiter=',')

W1 = np.loadtxt(f'W1_{corpus_name}.csv', delimiter=',')
W2 = np.loadtxt(f'W2_{corpus_name}.csv', delimiter=',')

# one update per batch (mini batch gradient descent)
J_batch_history = []
J_epoch_history = []
isample = 0


for iepoch in range(n_epochs):
    J_epoch = 0

    for ibatch, batch in enumerate(grouper(all_samples, batch_size)):
        dedw1_batch = np.zeros_like(W1)
        dedw2_batch = np.zeros_like(W2)
        J_batch = 0

        for sample in batch:
            # skip samples with no context words
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
        J_batch_history.append(J_batch)
        J_epoch += J_batch

    J_epoch /= (ibatch+1)
    J_epoch_history.append(J_epoch)
    print('iepoch={}, J_epoch={}'.format(iepoch, J_epoch))
