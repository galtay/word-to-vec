"""
Create a toy word2vec corpus from
https://www.kaggle.com/gjbroughton/start-trek-scripts
"""

import json
import re
import spacy

nlp = spacy.load('en_core_web_sm')
fname = 'all_scripts_raw.json'
dat = json.load(open(fname, 'r'))
tng = dat['TNG']
charptrn = '^[A-Z\]\[ ]+:'
tng_scripts = []


for episode_num, episode_text in tng.items():

    print('episode num: ', episode_num)

    episode_lines = episode_text.split('\n')
    episode_lines = [line for line in episode_lines if line]

    episode_lines = [re.sub(charptrn, '', line).strip() for line in episode_lines]
    episode_lines = [line for line in episode_lines if line]

    episode_text = ' '.join(episode_lines)
    doc = nlp(episode_text)

    sentences = [sent for sent in doc.sents]
    sentences = [
        [tok.text.lower() for tok in sent if tok.pos_ != 'PUNCT']
        for sent in sentences]
    sentences = [sent for sent in sentences if len(sent) > 1]


    tng_scripts.append({
        'episode_num': episode_num,
        'sentences': sentences,
    })


json.dump(tng_scripts, open('tng_scripts.json', 'w'))
