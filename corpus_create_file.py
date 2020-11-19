# %%
import pickle

with open('data/doc_labels_plenar.pkl', 'rb') as f:
    doc_labels_plenar = pickle.load(f)

doc_labels = [*doc_labels_plenar]

def gendocs(label):
    with open('data/corpus_clean/{}.txt'.format(label), "r") as text_file:
        return text_file.read()

# %%
with open('corpus_plenar.txt', 'w') as f:
    for label in doc_labels[:]:
        doc = gendocs(label)
        f.write(doc)
        f.write("\n")

# %%
from gensim.corpora.textcorpus import TextCorpus
d = TextCorpus('data/corpus_plenar.txt')

# %%
corpus = d.getstream()
next(corpus)

# %%
def gendocs(path):
    with open(path) as infile:
        for line in infile:
            yield line

d = gendocs('data/corpus_plenar.txt')
next(d)

# %%
for text in list(d)[:500]:
    print(text)

