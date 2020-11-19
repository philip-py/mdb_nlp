# %%
from gensim.models.wrappers import FastText
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim import utils
import gensim.models
import random
from germalemma import GermaLemma
import pickle
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
import spacy
from spacy.tokens import Doc, Span, Token
from tqdm import tqdm

# %%
with open('data/doc_labels_plenar.pkl', 'rb') as f:
    doc_labels_plenar = pickle.load(f)

# doc_labels = [*doc_labels_presse, *doc_labels_twitter, *doc_labels_plenar]

doc_labels = [*doc_labels_plenar]
def gendocs(label):
    with open('data/corpus_clean/{}.txt'.format(label), "r") as text_file:
        return text_file.read()

# %%
# save model in word2vec format
model = gensim.models.KeyedVectors.load('models/wv_plenar.model')

# %%
model.save_word2vec_format('embeddings/wiki.de.w2v')


# %%
# with open('/media/philippy/SSD/content_analysis/doc_labels_presse.pkl', 'rb') as f:
#     doc_labels_presse = pickle.load(f)
# with open('/media/philippy/SSD/content_analysis/doc_labels_twitter.pkl', 'rb') as f:
#     doc_labels_twitter = pickle.load(f)
# with open('/media/philippy/SSD/content_analysis/doc_labels_plenar.pkl', 'rb') as f:
#     doc_labels_plenar = pickle.load(f)

# doc_labels = [*doc_labels_presse, *doc_labels_twitter, *doc_labels_plenar]


# def gendocs(label):
#     with open('/media/philippy/SSD/corpus_clean/{}.txt'.format(label), "r", encoding='utf-8') as text_file:
#         return text_file.read()


# %%

lemmatizer = GermaLemma()


def lemma_getter(token):
    try:
        return lemmatizer.find_lemma(token.text, token.tag_)
    except:
        return token.lemma_

# doc_list = (gendocs(label) for label in doc_labels[:10])


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
            # print(sent)
            yield res
    # print(res)


nlp = spacy.load("de_core_news_lg")

# doc_labels = random.sample(doc_labels, 500)

# print(list(sentences_gen(doc_labels)))


# %%


class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __iter__(self):
        for sent in sentences_gen(doc_labels):
            # assume there's one document per line, tokens separated by whitespace
            yield sent


sentences = MyCorpus()

# %%
model = gensim.models.Word2Vec(sentences=tqdm(sentences), min_count=5, workers=7)

# %%
model.wv.most_similar(positive=['elite'], topn=100)

# %%
model.save('word_embedding_plenar.model')



####################
# %%
# create w2v format to export -> http://projector.tensorflow.org/
# model2 = gensim.models.Word2Vec.load("word_embedding_plenar.model")
model = gensim.models.Word2Vec.load("models/gnsm_w2v_plenar")

# %%
from gensim.models import Word2Vec, KeyedVectors
model.wv.save_word2vec_format('model', binary=False)



######################
# %%
model2.wv.most_similar(positive=['parlament'], negative=[''], topn=100)

# %%
model.wv.most_similar_cosmul(positive=['volk', 'ausländer'], negative=['deutsch'])

# model.wv.most_similar_to_given('volk', ['menschen', 'auto', 'parlament'])









# %%
import tempfile

with tempfile.NamedTemporaryFile(prefix='gensim-model-w2v', delete=False) as tmp:
    temporary_filepath = tmp.name
    model.save(temporary_filepath)
# %%


def hash(string):
    return 0


model_2 = gensim.models.Word2Vec(
    size=300, min_count=10, seed=122, workers=1, hashfxn=hash, sorted_vocab=1)
model_2.build_vocab(sentences)

# %%
# embedding_path = "/media/philippy/SSD/german.w2v"
embedding_path = "embeddings/wiki.de.w2v"
total_examples = model_2.corpus_count
model_2.intersect_word2vec_format(embedding_path, binary=False, lockf=1.0)
model_2.train(sentences, total_examples=total_examples, epochs=25)

# %%
# model_2.train(sentences, epochs=2)

# %%
print(model_2.most_similar('volk', topn=100))

# %%
model = gensim.models.Word2Vec(
    sentences=sentences, min_count=10, workers=7, seed=42)


# %%
nlp = spacy.load('de_core_news_lg')

wordList = []
vectorList = []
for key, vector in nlp.vocab.vectors.items():
    wordList.append(nlp.vocab.strings[key])
    vectorList.append(vector)

kv = WordEmbeddingsKeyedVectors(nlp.vocab.vectors_length)

kv.add(wordList, vectorList)

print(kv.most_similar('deutsch'))

# %%
glove_file = datapath('/media/philippy/SSD/vectors_glove.txt')
tmp_file = get_tmpfile("/media/philippy/SSD/test_word2vec.txt")
_ = glove2word2vec(glove_file, tmp_file)
model = KeyedVectors.load_word2vec_format(tmp_file)


# %%

# %%
save_word2vec_format('/media/philippy/')

# %%
print(model.most_similar('volk'))

# %%
# model.save("/media/philippy/SSD/pretrained_wikipedia_de.model")

# %%
# load word2vec model
# model = KeyedVectors.load_word2vec_format("/media/philippy/SSD/pretrained_wikipedia_de.model.vectors.npy", binary=False)
# model = gensim.models.KeyedVectors.load("/media/philippy/SSD/pretrained_wikipedia_de.model")
# model = gensim.models.KeyedVectors.load_word2vec_format("/media/philippy/SSD/pretrained_wikipedia_de.model")

# model = gensim.models.Word2Vec.load("/media/philippy/SSD/german.model")

model = gensim.models.KeyedVectors.load_word2vec_format(
    "/media/philippy/SSD/german.model", binary=True)

# %%
model.save_word2vec_format("/media/philippy/SSD/german.w2v")

# %%
model_2 = gensim.models.Word2Vec(size=300, min_count=1)
# model_2.build_vocab(sentences)
total_examples = model_2.corpus_count
model = KeyedVectors.load_word2vec_format(
    "/media/philippy/SSD/vectors_glove.txt", binary=False)
model_2.build_vocab([list(model.vocab.keys())], update=True)
# model_2.intersect_word2vec_format("glove.6B.300d.txt", binary=False, lockf=1.0)


# %%
model_w2v = KeyedVectors.load_word2vec_format(
    "/media/philippy/SSD/pretrained_wiki_de_w2v")


# %%
# model.build_vocab(sentences)
model.train(sentences, epochs=5)

# %%
print(model.most_similar('gut', topn=50))

# %%
model = KeyedVectors.load("/media/philippy/SSD/vectors.txt")

# %%
word_vectors = KeyedVectors.load(
    '/media/philippy/SSD/vectors.txt', binary=False)

# %%


# %%

model = FastText.load('/media/philippy/SSD/wiki.de.vec')

# %%
print(model.most_similar('volk', topn=50))

# %%
model = KeyedVectors.load_word2vec_format('/media/philippy/SSD/wiki.de.vec')

# %%
