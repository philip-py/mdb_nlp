# %%
from spacy.tokens import Doc, Span, Token
import spacy
import random
from germalemma import GermaLemma
import numpy as np
from gensim.utils import simple_preprocess
from gensim import models
from gensim import corpora
import pandas as pd
import pprint
import gensim
import pickle

# with open('doc_labels_presse.pkl', 'rb') as f:
# doc_labels_presse = pickle.load(f)
# with open('doc_labels_twitter.pkl', 'rb') as f:
# doc_labels_twitter = pickle.load(f)
with open('data/doc_labels_plenar.pkl', 'rb') as f:
    doc_labels_plenar = pickle.load(f)

# doc_labels = [*doc_labels_presse, *doc_labels_twitter, *doc_labels_plenar]

doc_labels = [*doc_labels_plenar]
print(len(doc_labels))


def gendocs(label):
    with open('data/corpus_clean/{}.txt'.format(label), "r") as text_file:
        return text_file.read()


# %%
# create gensim dict & BoW

lemmatizer = GermaLemma()

from src.d01_ana.analysis import load_data, gendocs
def lemma_getter(token):
    try:
        return lemmatizer.find_lemma(token.text, token.tag_)
    except:
        return token.lemma_

# doc_list = (gendocs(label) for label in doc_labels[:10])


def pipe(label):
    doc = nlp(gendocs(label))
    res = []

    for i, sent in enumerate(doc.sents):
        for j, token in enumerate(sent):
            Token.set_extension('lemma', getter=lemma_getter, force=True)
            if not token.is_punct and not token.is_digit and not token.is_space:
                tok = token._.lemma.lower()
                tok = tok.replace('.', '')
                res.append(tok)

    # print(res)
    return res

# sample = random.sample(doc_labels, 100)


nlp = spacy.load("de_core_news_lg")

docs = (pipe(label) for label in doc_labels)
tokens = [(token for token in doc) for doc in docs]
dictionary = corpora.Dictionary()


# %%
BoW_corpus = [dictionary.doc2bow(token, allow_update=True) for token in tokens]

# %%
dictionary.save('plenar_dict.pkl')

# %%
dictionary.keys()

# %%
tfidf = models.TfidfModel(BoW_corpus, smartirs='ntc')

# %%
# corpus = [dictionary.doc2bow(sent) for sent in tokens]
vocab_tf = {}
for i in BoW_corpus:
    for item, count in dict(i).items():
        if item in vocab_tf:
            vocab_tf[item] += count
        else:
            vocab_tf[item] = count


word_frequencies = [(dictionary[id], frequence)
                    for id, frequence in vocab_tf.items()]

df = pd.DataFrame(word_frequencies)

# %%


t = df[df[0].str.contains('zwang')]

# %%

# %%
tfidf.save('plenar_tfidf')

# %%

file_name = 'gnsm_bow.pkl'
pickle.dump(BoW_corpus, open(file_name, 'wb'))
# loaded_model = pickle.load(open(file_name, 'rb))


# %%

dictionary = pickle.load(open('gnsm_dict_all.pkl', 'rb'))
BoW_corpus = pickle.load(open('gnsm_bow.pkl', 'rb'))
# %%
