# %%
import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim import utils
import random
from germalemma import GermaLemma
import pickle
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
import spacy
from spacy.tokens import Doc, Span, Token
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import hashlib
import logging
from spacy.lang.de.stop_words import STOP_WORDS
import sys
import matplotlib.pyplot as plt

%env PYTHONHASHSEED=0
sns.set_style('darkgrid')

logging.basicConfig(filename='w2v.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""
    def __init__(self, sample):
        self.docs = sample

    def __iter__(self):
        for sent in sentences_gen(self.docs):
            yield sent

class LossLogger(CallbackAny2Vec):
    '''Get the Loss after every epoch and log it to a file'''
    def __init__(self, name, i):
        self.epoch=1
        self.last_cum_loss = 0
        self.last_epoch_loss = 0
        self.losses = []
        self.best_loss = 1e15
        self.best_model = None
        self.name = name
        self.iteration = i

    def on_epoch_end(self, model):

        if not self.name:
            self.name = 'ALL'
        cum_loss=model.get_latest_training_loss()
        logging.info('Cumulative Loss after epoch {}: {}'.format(self.epoch, cum_loss))
        logging.info('Cumulative Loss last epoch : {}'.format(self.last_cum_loss))
        this_epoch_loss = cum_loss - self.last_cum_loss
        loss_diff = this_epoch_loss - self.last_epoch_loss
        self.losses.append(this_epoch_loss)

        logging.info('Loss in epoch {}: {}'.format(self.epoch, this_epoch_loss))
        logging.info('Loss in last epoch: {}'.format(self.last_epoch_loss))
        logging.info('Loss difference since last epoch: {}'.format(loss_diff))
        print(f'Loss difference: {loss_diff}')

        if this_epoch_loss < self.best_loss:
            self.best_model = model
            self.best_loss = this_epoch_loss
            logging.info('saving best model in epoch {} with loss {}'.format(self.epoch, this_epoch_loss))
            model.save(f'w2v_models/emb_{self.name}_{self.iteration}.model')

        self.epoch=self.epoch+1
        self.last_cum_loss = cum_loss
        self.last_epoch_loss = this_epoch_loss

        if this_epoch_loss == 0.0:
            # sys.exit()
            # sns.lineplot(data=self.losses)
            # plt.show()
            raise EndOfTraining()

class EndOfTraining(Exception):
    pass

def load_data(party=None):
    with open("data/doc_labels_plenar.pkl", "rb") as f:
        doc_labels_plenar = pickle.load(f)

    # doc_labels = [*doc_labels_presse, *doc_labels_twitter, *doc_labels_plenar]

    doc_labels = [*doc_labels_plenar]

    if not party:
        return doc_labels

    df = pd.read_json('data/plenar_meta.json', orient='index')
    res = df.loc[df.party == party].index.values
    doc_labels = [i.split('.txt')[0] for i in res]
    # return random.sample(doc_labels, 1)
    return doc_labels

def gendocs(label):
    with open("data/corpus_clean/{}.txt".format(label), "r") as text_file:
        return text_file.read()

def lemma_getter(token):
    try:
        return lemmatizer.find_lemma(token.text, token.tag_)
    except:
        return token.lemma_

def sentences_gen(labels):
    for label in labels:
        doc = nlp(gendocs(label))
        for i, sent in enumerate(doc.sents):
            res = []
            for j, token in enumerate(sent):
                Token.set_extension('lemma', getter=lemma_getter, force=True)
                if not token.is_punct and not token.is_digit and not token.is_space:
                    tok = token._.lemma.lower()
                    tok = tok.replace('.', '')
                    res.append(tok)
            res = [word for word in res if not word in STOP_WORDS]
            yield res


def main(party, iter):

    sample = load_data(party)
    # sample = [load_data(party)[100]]
    print('Number of documents: {}'.format(len(sample)))
    sentences = MyCorpus(sample)

    # model params
    model = Word2Vec(
    alpha=0.0025, min_alpha=0.00001, vector_size=300, min_count=50, epochs=300, seed=42, workers=8, hashfxn=hash, sorted_vocab=1, sg=1, hs=1, negative=0, sample=1e-5)

    model.build_vocab(sentences)

    # intersect
    pre = KeyedVectors.load('wiki.model')
    res = intersect(pre, model)
    del pre

    model.wv.add_vectors(range(model.wv.vectors.shape[0]), res, replace=True)

    # ASSERT?

    total_examples = model.corpus_count

    for i in range(iter):
        try:
            seeds = random.choices(range(1_000_000), k=iter)
            seed = seeds[i]
            print(f'Seed in iteration {i}: {seed}')
            model.seed = seed
            loss_logger = LossLogger(party, i)

            # train
            model.train(sentences, total_examples=total_examples, epochs=model.epochs, compute_loss=True, callbacks=[loss_logger])

        except EndOfTraining:
            print(f'End of Iteration: {i}')


# %%
if __name__ == "__main__":

    party = None
    iter = 3

    lemmatizer = GermaLemma()
    nlp = spacy.load("de_core_news_lg")

    main(party, iter)

    # %%
    res = merge_embeddings(load_models(party, iter))

    # %%














# %%
    # afd:
    # model = Word2Vec(
    # alpha=0.0025, min_alpha=0.00001, vector_size=300, min_count=10, epochs=300, seed=42, workers=8, hashfxn=hash, sorted_vocab=1, sg=1, hs=1, negative=0, sample=0)


# %%
# measure distance between words:
# cosine similarity
import numpy
from scipy import spatial
res = dict()
for word in tuned.wv.vocab:
    if word not in model.wv.vocab:
        pass
    else:
        # cosine_similarity = numpy.dot(model[word], afd[word])/(numpy.linalg.norm(model[word])* numpy.linalg.norm(afd[word]))
        cosine_similarity = 1 - spatial.distance.cosine(model[word], tuned[word])
        res[word] = cosine_similarity

# %%
import heapq
from operator import itemgetter

n = 100

minitems = heapq.nsmallest(n, res.items(), key=itemgetter(1))
maxitems = heapq.nlargest(n , res.items(), key=itemgetter(1))



# %%
# eucledian distance
res = dict()
for word in afd.wv.vocab:
    if word not in model.wv.vocab:
        pass
    else:
    # w2c[item]=model.wv.vocab[item].count
        dist = numpy.linalg.norm(model[word] - afd[word])
        res[word] = dist

# %%
# in relation to frequency?
# get term frequency in corpus
tf = dict()
for word in model.wv.vocab:
    tf[word] = model.wv.vocab[word].count / model.corpus_total_words

# %%
n = 100
maxitems = heapq.nlargest(n , tf.items(), key=itemgetter(1))
check = [i[0] for i in minitems]
res_freq = [i for i in check if afd.wv.vocab[i].count > 20]
res_freq = [i for i in res_freq if afd.wv.vocab[i].count < 300]
res_freq


# %%
# tensor projection:
# How to use
# Convert your word-vector with this script (for example, we’ll use model from gensim-data)

# python -m gensim.downloader -d glove-wiki-gigaword-50  # download model in word2vec format

# save for tf-projector:
# afd.wv.save_word2vec_format('model', binary=False)

# python -m gensim.scripts.word2vec2tensor -i embeddings_plenar_afd.model -o model_afd

# Open http://projector.tensorflow.org/

# Click “Load Data” button from the left menu.

# Select “Choose file” in “Load a TSV file of vectors.” and choose “/tmp/my_model_prefix_tensor.tsv” file.

# Select “Choose file” in “Load a TSV file of metadata.” and choose “/tmp/my_model_prefix_metadata.tsv” file.

# ???

# PROFIT!

# jupyter nbconvert results.ipynb --TagRemovePreprocessor.remove_input_tags="{'hide_input'}" --no-prompt
