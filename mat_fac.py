"""
Pedagogical implementation of word vectors from matrix facotrization.
"""
from collections import Counter
import json
import string
import numpy as np
import os
from scipy import sparse
from scipy.sparse import linalg
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
TOL = 1.0e-7

# corpora are pre-tokenized in json format
# each corpus is a list of lists of words
# vocabs are lists of word, count pairs sorted by frequency
#-------------------------------------------

#corpus_name = 'space_docs_big_idea'
corpus_name = 'tng_scripts'

corpus_path = os.path.join('corpora', corpus_name)
corpus_file = os.path.join(corpus_path, 'corpus.json')
vocab_file = os.path.join(corpus_path, 'vocab.json')

with open(corpus_file, 'r') as fp:
    corpus = json.load(fp)
with open(vocab_file, 'r') as fp:
    vocab = json.load(fp)


# create word to index mapping and inverse
#-------------------------------------------
tok2index = {}
index2tok = {}
for index, (tok, count) in enumerate(vocab):
    tok2index[tok] = len(tok2index)
index2tok = {index:tok for tok, index in tok2index.items()}



# note add dynammic window hyperparameter
back_window = 2
front_window = 2
skipgram_counts = Counter()
for isequence, sequence in enumerate(corpus):
    for ifocus, focus_token in enumerate(sequence):
        icontext_min = max(0, ifocus - back_window)
        icontext_max = min(len(sequence) - 1, ifocus + front_window)
        icontexts = [ii for ii in range(icontext_min, icontext_max + 1) if ii != ifocus]
        for icontext in icontexts:
            skipgram = (sequence[ifocus], sequence[icontext])
            skipgram_counts[skipgram] += 1
    if isequence % 200000 == 0:
        print(f'finished {isequence/len(corpus):.2%} of corpus')

print('done')
print('number of skipgrams: {}'.format(len(skipgram_counts)))
print('most common: {}'.format(skipgram_counts.most_common(10)))


# word-word count matrix
#=======================================================================
row_indexs = []
col_indexs = []
dat_values = []
ii = 0
for (tok1, tok2), sg_count in skipgram_counts.items():
    ii += 1
    if ii % 1000000 == 0:
        print(f'finished {ii/len(skipgram_counts):.2%} of skipgrams')
    tok1_index = tok2index[tok1]
    tok2_index = tok2index[tok2]

    row_indexs.append(tok1_index)
    col_indexs.append(tok2_index)
    dat_values.append(sg_count)

wwcnt_mat = sparse.csr_matrix((dat_values, (row_indexs, col_indexs)))
wwcnt_norm_mat = normalize(wwcnt_mat, norm='l2', axis=1)
print('done')


embedding_size = 50
uu, ss, vv = linalg.svds(wwcnt_norm_mat, embedding_size)


def ww_sim(word, mat, topn=10):
    """Calculate topn most similar words to word"""
    index = tok2index[word]
    if isinstance(mat, sparse.csr_matrix):
        v1 = mat.getrow(index)
    else:
        v1 = mat[index:index+1, :]
    sims = cosine_similarity(mat, v1).flatten()
    sindexs = np.argsort(-sims)
    sim_word_scores = [(index2tok[sindex], sims[sindex]) for sindex in sindexs[0:topn]]
    return sim_word_scores
