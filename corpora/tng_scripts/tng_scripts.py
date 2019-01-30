from collections import Counter
import json
import string
import numpy as np

class TngScripts:

    def __init__(self, file_path):
        with open(file_path, 'r') as fp:
            self._scripts = json.load(fp)

    def iter_sentences(self):
        for episode in self._scripts:
            for sent in episode['sentences']:
                yield sent

    def iter_target_context(self, back_window=2, front_window=2, shuffle=True):

        episode_indices = np.arange(len(self._scripts))
        if shuffle:
            np.random.shuffle(episode_indices)

        for episode_index in episode_indices:

            sentences = self._scripts[episode_index]['sentences']
            sentence_indices = np.arange(len(sentences))
            if shuffle:
                np.random.shuffle(sentence_indices)

            for sentence_index in sentence_indices:
                sentence = sentences[sentence_index]
                for target_indx, target_word in enumerate(sentence):

                    min_context_indx = max(0, target_indx - back_window)
                    max_context_indx = min(len(sentence)-1, target_indx + front_window)

                    context_indices = [
                        indx for indx in range(min_context_indx, max_context_indx + 1)
                        if indx != target_indx]
                    context_words = [sentence[indx] for indx in context_indices]

                    yield (target_word, context_words)


    def iter_target_context_batch(self, back_window=2, front_window=2, batch_size=10, shuffle=True):

        tc_iter = self.iter_target_context(
            back_window=back_window,
            front_window=front_window,
            shuffle=shuffle)

        keep_going = True
        while keep_going:
            batch = []
            for ii in range(batch_size):
                try:
                    batch.append(next(tc_iter))
                except StopIteration:
                    keep_going = False
                    pass
            yield batch


if __name__ == '__main__':

    file_path = 'tng_scripts.json'
    corpus = TngScripts(file_path)
    sentences = list(corpus.iter_sentences())

    # write json corpus
    corpus_file = 'corpus.json'
    with open(corpus_file, 'w') as fp:
        json.dump(sentences, fp)

    # write vocabulary
    word_counter = Counter([word for sent in sentences for word in sent])
    vocab_file = 'vocab.json'
    with open(vocab_file, 'w') as fp:
        json.dump(word_counter.most_common(), fp)
