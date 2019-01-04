import string
import numpy as np

class SpaceDocsBigIdea:

    def __init__(self, file_path):
        with open(file_path, 'r') as fp:
            text = fp.read()
        for punc in string.punctuation:
            text = text.replace(punc, '')
        text = text.split('\n')
        text = [sent.lower() for sent in text if sent != '']
        text = [sent.split() for sent in text]
        self._text = text

        self._list_of_words = [word for sent in text for word in sent]


    def get_text(self):
        return self._text


    def iter_sentences(self):
        for sent in self._text:
            yield sent


    def iter_target_context(self, back_window=2, front_window=2, shuffle=True):

        sentence_indices = np.arange(len(self._text))
        if shuffle:
            np.random.shuffle(sentence_indices)

        for sentence_index in sentence_indices:
            sentence = self._text[sentence_index]
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

    file_path = 'space_docs_big_idea.txt'
    sdbi = SpaceDocsBigIdea(file_path)

    for batch in sdbi.iter_target_context_batch(shuffle=False, batch_size=12):
        target = [el[0] for el in batch]
        print(target)
        print()
