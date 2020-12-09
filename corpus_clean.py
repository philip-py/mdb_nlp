# %%
from tmtoolkit.corpus import Corpus
from tmtoolkit.preprocess import TMPreproc
import pandas as pd
from src.d00_utils.del_chars import del_chars
import re
from pprint import pprint
import string
import pickle
from tqdm import tqdm

# %%
PATH = 'data/corpus'
corpus = Corpus.from_folder(PATH + '/plenar', encoding='utf8')
# corpus = Corpus.from_folder(PATH + '/plenar')
# corpus.add_folder(PATH + '/presse')
# corpus.add_folder(PATH + '/twitter')

doc_labels = corpus.get_doc_labels(sort=True)

# %%
table_umlauts = {"ÃŸ": "ß", "ãÿ": "ß", "ã¤": "ä", "ã¼": "ü", "ã¶": "ö", 'Ã„': 'Ä', "Ãœ": "Ü", "Ã–": "Ö", 'â‚¬': '€'}

table_chars = {';': '.', '$': '', '?': '.', '!': '.', ':':'.'}

# phrases = {'teilentweetPrint': '', 'Current Page': '', 'Pressekontakt .   CDUCSU  BundestagsfraktionPressestelleTelefon .   030 22752360Fax .       030 22756660Internet .  http . www . cducsu . deEmail .  pressestellecducsu . de OriginalContent von .  CDUCSU  Bundestagsfraktion, übermittelt durch news aktuell': '', }

def repl_phrases(doc):
    for k, v in phrases.items():
        doc = doc.replace(k,v)
    return doc

def repl_umlauts(doc):
    for k, v in table_umlauts.items():
        doc = doc.replace(k,v)
    return doc

def repl_chars(doc):
    for k, v in table_chars.items():
        doc = doc.replace(k, v)
    return doc

def repl_nl(doc):
    doc = doc.replace(r'\n', "")
    return doc

def repl_last(doc):
    doc = doc.replace('-', ' ')
    return doc

def repl_dot(doc):
    doc = doc.replace('.', ' . ')
    return doc

def fix_spaces(doc):
    doc = ' '.join(doc.split())
    return doc

corpus.apply(lambda x: repl_umlauts(x))
corpus.apply(lambda x: repl_chars(x))
# corpus.apply(lambda x: repl_nl(x))

corpus.replace_characters(del_chars)

# correct contractions
pttrn_contraction_ws = re.compile(r'(\w+)(\s+)(-\w+)')
corpus.apply(lambda t: pttrn_contraction_ws.sub(lambda m: m.group(1) + m.group(3), t))

corpus.apply(lambda x: repl_last(x))
corpus.apply(lambda x: repl_dot(x))
# corpus.apply(lambda x: repl_phrases(x))

corpus.apply(fix_spaces)

# %%
# delete special chars in tweets:
left = corpus.unique_characters - set(string.printable)
umlauts = ['ä', 'ü', 'ö', 'Ä', 'Ö', 'Ü', 'ß']
for um in umlauts:
    left.discard(um)
left_dict = {d: None for d in left}

corpus.replace_characters(left_dict)

# %%
print('these non-ASCII characters are left:')
pprint(corpus.unique_characters - set(string.printable))


# %%
# check
import random
r = random.sample(doc_labels, 50)

# %%
for doc in r:
    print()
    print(doc)
    print(len(corpus[doc]))
    # s = corpus[doc]
    # s = ' '.join(s.split())
    print(corpus[doc])


# %%
# write labels to disk
with open('data/doc_labels_plenar.pkl', 'wb') as f:
    pickle.dump(doc_labels, f)

# %%
# write corpus to disk
for label in tqdm(doc_labels):
    with open('data/corpus_clean/{}.txt'.format(label), "w", encoding='utf8') as text_file:
        text_file.write(corpus[label])

# %%
