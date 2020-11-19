# %%
import pandas as pd

# %%
sent = pd.read_csv('dict/SentDict.csv')
# %%
sent_neg = pd.read_csv('dict/SentDictNeg.csv')
# %%
# list of pos sentiment:
pos = [x.strip() for x in sent.loc[sent.sentiment == 1, ['feature']]['feature'].tolist()]

neg = [x.strip() for x in sent.loc[sent.sentiment == -1, ['feature']]['feature'].tolist()]
# %%
neg
# %%
