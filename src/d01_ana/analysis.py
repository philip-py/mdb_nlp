# %%
import sys
import pickle
import random
import spacy
import logging
import hashlib
import glob
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log
from pathlib import Path
from collections import Counter
from germalemma import GermaLemma
from tqdm import tqdm
from spacy.tokens import Doc, Span, Token
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.lang.de.stop_words import STOP_WORDS
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim import utils
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import tfidfmodel

# %%
class ContentAnalysis:
    def __init__(self, model, dict_folder, directory):

        self.nlp = spacy.load(model)
        self.folder = directory

        Path(f'res_ca/{self.folder}/').mkdir(parents=False, exist_ok=False)

        # import all dicts
        # elite
        df_elite = pd.read_csv(f"{dict_folder}/elite_dict.csv")
        self.elite = set(df_elite[df_elite.type != "elite_noneg"]["feature"].tolist())
        self.elite_noneg = set(
            df_elite[df_elite.type == "elite_noneg"]["feature"].tolist()
        )

        # people
        df_people = pd.read_csv(f"{dict_folder}/people_dict.csv")
        self.people = set(df_people[df_people.type == "people"]["feature"].tolist())
        self.people_ordinary = set(
            df_people[df_people.type == "people_ordinary"]["feature"].tolist()
        )
        self.attr_ordinary = set(
            df_people[df_people.type == "attr_ordinary"]["feature"].tolist()
        )
        self.people_ger = set(
            df_people[df_people.type == "people_ger"]["feature"].tolist()
        )
        self.attr_ger = set(df_people[df_people.type == "attr_ger"]["feature"].tolist())

        # list of sentiment:
        sent = pd.read_csv(f"{dict_folder}/SentDict.csv")
        self.positiv = set(
            [
                x.strip()
                for x in sent.loc[sent.sentiment == 1, ["feature"]]["feature"].tolist()
            ]
        )
        self.negativ = set(
            [
                x.strip()
                for x in sent.loc[sent.sentiment == -1, ["feature"]]["feature"].tolist()
            ]
        )

        # custom lemmatizer
        self.lemmatizer = GermaLemma()

    def analyze(self, label, text, window_size):
        def lemma_getter(token):
            try:
                return self.lemmatizer.find_lemma(token.text, token.tag_)
            except:
                return token.lemma_

        def is_neg_getter(token):
            if token._.lemma in self.negativ:
                check = list(token.children)
                node = token
                while node.head:
                    seen = node
                    if seen == node.head:
                        break
                    else:
                        check.append(node)
                        check.extend(list(node.children))
                        node = seen.head

                attr = [child for child in check if child.dep_ == "ng"]
                if attr:
                    return False
                else:
                    return True
                return True

            elif token._.lemma in self.positiv:
                check = list(token.children)
                node = token
                while node.head:
                    seen = node
                    if seen == node.head:
                        break
                    else:
                        check.append(node)
                        check.extend(list(node.children))
                        node = seen.head
                attr = [child for child in check if child.dep_ == "ng"]
                if attr:
                    return True
                else:
                    return False

        def is_pos_getter(token):
            if token._.lemma in self.positiv:
                return True

            elif token._.lemma in self.negativ:
                check = list(token.children)
                node = token
                while node.head:
                    seen = node
                    if seen == node.head:
                        break
                    else:
                        check.append(node)
                        check.exten(list(node.children))
                        node = seen.head
                attr = [child for child in check if child.dep_ == "ng"]
                if attr:
                    return True
                else:
                    return False

        def is_neg_elite(token):

            if token._.is_elite_noneg:
                info_token = (token._.lemma, None)
                token._.info = info_token
                return True

            elif token._.is_elite:
                check = list(token.children)
                # if token.head:
                #     check.append(token.head)
                node = token
                while node.head:
                    #     seen = node
                    #     if seen == node.head:
                    #         break
                    #     else:
                    #         check.append(node)
                    #         node = seen.head

                    seen = node
                    for t in node.ancestors:
                        if t.dep_ == "conj":
                            break
                        for tok in t.children:
                            if tok.dep_ == "pd":
                                check.append(tok)
                            if tok.dep_ == "mo":
                                check.append(tok)
                            if tok.dep_ == "oa":
                                check.append(tok)
                            if tok.dep_ == "oa2":
                                check.append(tok)
                            if tok.dep_ == "og":
                                check.append(tok)
                            if tok.dep_ == "da":
                                check.append(tok)
                            if tok.dep_ == "op":
                                check.append(tok)
                            if tok.dep_ == "cc":
                                check.append(tok)
                            # elif tok.dep_ == 'oprd':
                            #     check.append(tok)
                            # elif tok.dep_ == 'attr':
                            #     check.append(tok)
                    check.append(node)
                    node = seen.head
                    if seen == node.head:
                        break

                attr = set([child._.lemma.lower() for child in check if child._.is_neg])
                if attr:
                    info_token = (token._.lemma, attr)
                    token._.info = info_token
                    return True
                else:
                    return False
                # return any([True for child in check if child._.lemma.lower() in negativ])
            else:
                return False

        def is_volk(token):

            # if token.pos_ == 'NOUN' or token.pos_ == 'PROPN':

            check = list(token.children)

            if token._.lemma.lower() in self.people:
                info_token = (token._.lemma, None)
                token._.info = info_token
                return True

            elif token._.lemma.lower() in self.people_ordinary:
                attr = set(
                    [
                        child._.lemma.lower()
                        for child in check
                        if child._.lemma.lower() in self.attr_ordinary
                    ]
                )
                if attr:
                    info_token = (token._.lemma, attr)
                    token._.info = info_token
                    return True
                else:
                    return False

            elif token._.lemma.lower() in self.people_ger:
                attr = set(
                    [
                        child._.lemma.lower()
                        for child in check
                        if child._.lemma.lower() in self.attr_ger
                    ]
                )
                if attr:
                    info_token = (token._.lemma, attr)
                    token._.info = info_token
                    return True
                else:
                    return False

            else:
                return False

        all_sents = []

        res_dict = {
            "doc": label,
            "len": None,
            "pop": False,
            "volk": 0,
            "elite": 0,
            "sents": None,
        }
        # doc = nlp(gendocs(label))
        doc = self.nlp(text)
        hits = {"volk": [], "volk_text": [], "elite": [], "elite_text": [], "attr": []}

        for i, sent in enumerate(doc.sents):
            for j, token in enumerate(sent):

                # is_negation_getter = lambda token: token._.lemma.lower() in self.negation
                is_elite_getter = lambda token: token._.lemma.lower() in self.elite
                is_elite_noneg_getter = (
                    lambda token: token._.lemma.lower() in self.elite_noneg
                )

                Token.set_extension("info", default=None, force=True)
                Token.set_extension("lemma", getter=lemma_getter, force=True)
                # Token.set_extension('is_negation', getter=is_negation_getter, force=True)
                Token.set_extension("is_neg", getter=is_neg_getter, force=True)
                Token.set_extension("is_pos", getter=is_pos_getter, force=True)
                Token.set_extension("is_elite", getter=is_elite_getter, force=True)
                Token.set_extension(
                    "is_elite_noneg", getter=is_elite_noneg_getter, force=True
                )

                is_volk_getter = lambda token: is_volk(token)
                is_neg_elite_getter = lambda token: is_neg_elite(token)

                Token.set_extension("is_volk", getter=is_volk_getter, force=True)
                Token.set_extension(
                    "is_neg_elite", getter=is_neg_elite_getter, force=True
                )

                if token._.is_volk:
                    hits["volk"].append(token._.lemma)
                    hits["volk_text"].append(token.text)

                if token._.is_neg_elite:
                    hits["elite"].append(token._.lemma)
                    hits["elite_text"].append(token.text)
                    hits["attr"].append(token._.info)
                    all_sents.append(sent)

                # Token.set_extension('is_pos_volk', getter=is_pos_volk_getter_func, force=True)


        matcher = Matcher(self.nlp.vocab)
        pattern = [{"_": {"is_neg_elite": True}}]
        matcher.add("text", None, pattern)
        matches = matcher(doc)
        has_pop = set()
        has_pop_ = set()
        # tokens_pop = []
        info = []
        for match_id, start, end in matches:
            span = doc[start - window_size : end + window_size]

            for token in span:
                if token._.is_volk:

                    # info.append(doc[start]._.info)
                    # info.append(token._.info)

                    # tokens_pop.append(doc[start]._.lemma)
                    # tokens_pop_all.append((doc[start]._.lemma, list(doc[start].children), token._.lemma))
                    # tokens_pop.append(token._.lemma)
                    sentence_start = span[0].sent.start
                    sentence_end = span[-1].sent.end
                    has_pop.add(doc[sentence_start:sentence_end].text)
                    has_pop_.add(doc[sentence_start:sentence_end])

        for i, sent in enumerate(has_pop_):
            info.append([[], []])
            for token in sent:
                if token._.is_volk:
                    info[i][0].append(token._.info)
                if token._.is_neg_elite:
                    info[i][1].append(token._.info)

        c_volk = Counter(([token._.is_volk for token in doc]))
        c_neg_elite = Counter(([token._.is_neg_elite for token in doc]))
        # tokens_pop_counter = Counter(tokens_pop)

        if has_pop:
            res_dict["pop"] = True
        res_dict["doc"] = label
        res_dict["sents"] = set(has_pop)
        res_dict["num_pop"] = len(has_pop)
        res_dict["elite"] = c_neg_elite[True]
        res_dict["volk"] = c_volk[True]
        res_dict["len"] = len(doc)
        res_dict["volk_lemma"] = hits["volk"]
        res_dict["volk_text"] = hits["volk_text"]
        res_dict["elite_lemma"] = hits["elite"]
        res_dict["elite_text"] = hits["elite_text"]
        res_dict["elite_attr"] = hits["attr"]
        res_dict["volk_counter"] = Counter(hits["volk"])
        res_dict["elite_counter"] = Counter(hits["elite"])
        # res_dict['lemma_pop'] = tokens_pop
        # res_dict['lemma_pop_count'] = tokens_pop_counter
        # res_dict['pop_all'] = tokens_pop_all
        res_dict["hits_pop"] = info
        # res.append(res_dict)
        return res_dict


class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __init__(self, sample):
        self.docs = sample

    def __iter__(self):
        for sent in sentences_gen(self.docs):
            yield sent


class LossLogger(CallbackAny2Vec):
    """Get the Loss after every epoch and log it to a file"""

    def __init__(self, party, i, directory):
        self.epoch = 1
        self.last_cum_loss = 0
        self.last_epoch_loss = 0
        self.losses = []
        self.best_loss = 1e15
        self.best_model = None
        self.name = party
        self.iteration = i
        self.folder = directory

    def on_epoch_end(self, model):

        cum_loss = model.get_latest_training_loss()
        logging.info("Cumulative Loss after epoch {}: {}".format(self.epoch, cum_loss))
        logging.info("Cumulative Loss last epoch : {}".format(self.last_cum_loss))
        this_epoch_loss = cum_loss - self.last_cum_loss
        loss_diff = this_epoch_loss - self.last_epoch_loss
        self.losses.append(this_epoch_loss)

        logging.info("Loss in epoch {}: {}".format(self.epoch, this_epoch_loss))
        logging.info("Loss in last epoch: {}".format(self.last_epoch_loss))
        logging.info("Loss difference since last epoch: {}".format(loss_diff))
        print(f"Epoch: {self.epoch} | Loss: {this_epoch_loss}")
        print(f"Loss difference: {loss_diff}")

        if this_epoch_loss < self.best_loss:
            self.best_model = model
            self.best_loss = this_epoch_loss
            logging.info(
                "saving best model in epoch {} with loss {}".format(
                    self.epoch, this_epoch_loss
                )
            )
            model.save(f"res_da/{self.folder}/emb_{self.name}_{self.iteration}.model")

        self.epoch = self.epoch + 1
        self.last_cum_loss = cum_loss
        self.last_epoch_loss = this_epoch_loss

        if this_epoch_loss == 0.0:
            # sys.exit()
            sns.lineplot(data=self.losses)
            plt.show()
            raise EndOfTraining()


class EndOfTraining(Exception):
    pass


def populism_score_main(df, dictionary, idf_weight):
    df['scores'] = df.apply(lambda row: compute_score_per_category(row, dictionary, idf_weight), axis=1)

    # seperate into columns:
    df['score'] = df.apply(lambda row: sum(row.scores), axis=1)
    df['score_volk'] = df.apply(lambda row: row.scores[0], axis=1)
    df['score_elite'] = df.apply(lambda row: row.scores[1], axis=1)
    df['score_attr'] = df.apply(lambda row: row.scores[2], axis=1)

    return df


def compute_score_from_counts(counts, doclen, dictionary, idf_weight):
    # number docs should be constant!
    scores = []
    for term, n in counts.items():
        score = compute_idf(term, dictionary)
        scores.append((score**idf_weight) * n)
    res = sum(scores) / log(doclen+10, 10)
    # res = sum(scores)
    return res


def compute_score_per_term(term, doclen, dictionary, idf_weight):
    score = compute_idf(term, dictionary) ** idf_weight
    res = score / log(doclen+10, 10)
    return res


def compute_score_per_category(row, dictionary, idf_weight):
    if row['pop'] == True:
        volk = compute_score_from_counts(row.volk_counter, row.len, dictionary, idf_weight)
        elite = compute_score_from_counts(row.elite_counter, row.len, dictionary, idf_weight)
        attr = compute_score_from_counts(row.attr_counter, row.len, dictionary, idf_weight)
    else:
        volk = 0.0
        elite = 0.0
        attr = 0.0
    return (volk, elite, attr)


def compute_idf(term, dictionary, idf_weight=None):
    df = dictionary.dfs[dictionary.token2id[term.lower()]]
    if idf_weight:
        return tfidfmodel.df2idf(df, dictionary.num_docs, log_base=2.0, add=1.0)**idf_weight
    else:
        score = tfidfmodel.df2idf(df, dictionary.num_docs, log_base=2.0, add=1.0)
        return score


def load_results_content_analysis(folder):
    """load dataframe from results folder"""
    all_files = glob.glob(f"{folder}/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        df.drop(columns = 'Unnamed: 0', inplace=True)
        li.append(df)
    dfog = pd.concat(li, axis=0, ignore_index=True)
    # dfog = pd.read_csv(filename)
    dfval = pd.read_json("data/plenar_meta.json", orient="index")
    # keep for future
    # dfval_2 = pd.read_json('/media/philippy/SSD/data/ma/corpus/presse_meta.json', orient='index')
    # dfval_3 = pd.read_json('/media/philippy/SSD/data/ma/corpus/twitter_meta.json', orient='index')
    # dfval = dfval_1.append([dfval_2, dfval_3])
    dfval["doc"] = dfval.index
    dfval["doc"] = dfval.doc.apply(lambda x: x.split(".")[0])
    # fix timestamps
    df = dfval.copy()
    df["date"] = df.datum
    df["date"] = pd.to_datetime(df["date"], unit="ms", errors="ignore")
    dfval = df
    # merge results and meta
    dfs = dfog.merge(dfval.loc[:, ["date", "party", "doc", "name_res", "gender", "election_list", "education", "birth_year"]], how="left", on="doc")
    dfs = dfs.set_index("date").loc["2013-10-01":"2020-01-01"]
    dfs["date"] = dfs.index
    # eval strings
    dfs["elite_attr"] = dfs.apply(lambda row: eval(str(row.elite_attr)), axis=1)
    dfs["volk_counter"] = dfs.apply(lambda row: eval(str(row.volk_counter)), axis=1)
    dfs["elite_counter"] = dfs.apply(lambda row: eval(str(row.elite_counter)), axis=1)
    dfs['attr_counter'] = dfs.apply(lambda row: Counter([term for i in row.elite_attr for term in (i[1] if i[1] is not None else [])]), axis=1)
    dfs["hits_pop"] = dfs.apply(lambda row: eval(str(row.hits_pop)), axis=1)
    # add type and opposition
    dfs["typ"] = dfs["doc"].apply(lambda row: row.split("_")[0])
    dfs["opp"] = dfs.apply(lambda row: isopp(row), axis=1)
    return dfs




def isopp(row):
    if row.party in ["CDU", "SPD", "CSU"]:
        return "not_opp"
    else:
        return "opp"


def merge_meta(df):
    dfval = pd.read_json("data/plenar_meta.json", orient="index")
    # dfval_2 = pd.read_json('/media/philippy/SSD/data/ma/corpus/presse_meta.json', orient='index')
    # dfval_3 = pd.read_json('/media/philippy/SSD/data/ma/corpus/twitter_meta.json', orient='index')
    # dfval = dfval_1.append([dfval_2, dfval_3])

    dfval["doc"] = dfval.index
    dfval["doc"] = dfval.doc.apply(lambda x: x.split(".")[0])

    dfval["date"] = dfval.datum
    dfval["date"] = pd.to_datetime(dfval["date"], unit="ms", errors="ignore")

    dfs = df.merge(dfval, how="left", on="doc")

    return dfs


def load_data(party):
    with open("data/doc_labels_plenar.pkl", "rb") as f:
        doc_labels_plenar = pickle.load(f)

    # doc_labels = [*doc_labels_presse, *doc_labels_twitter, *doc_labels_plenar]

    doc_labels = [*doc_labels_plenar]

    if party == "all":
        return doc_labels

    df = pd.read_json("data/plenar_meta.json", orient="index")
    res = df.loc[df.party == party].index.values
    doc_labels = [i.split(".txt")[0] for i in res]
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
    lemmatizer = GermaLemma()
    nlp = spacy.load("de_core_news_lg")
    for label in labels:
        doc = nlp(gendocs(label))
        for i, sent in enumerate(doc.sents):
            res = []
            for j, token in enumerate(sent):
                Token.set_extension("lemma", getter=lemma_getter, force=True)
                if not token.is_punct and not token.is_digit and not token.is_space:
                    tok = token._.lemma.lower()
                    tok = tok.replace(".", "")
                    res.append(tok)
            res = [word for word in res if not word in STOP_WORDS]
            yield res


def hash(w):
    return int(hashlib.md5(w.encode("utf-8")).hexdigest()[:9], 16)


def intersect(pre, new):
    """
    intersect embeddings weights
    pre -> pre-trained embeddings
    """
    res = np.zeros(new.wv.vectors.shape)
    for i, word in enumerate(new.wv.index_to_key):
        if pre.has_index_for(word):
            res[i] = pre.get_vector(word)
        else:
            res[i] = new.wv.get_vector(word)
    return res


def merge_embeddings(models):
    """
    models -> List of models from Word2Vec.load('path')
    Returns model with average weights of all embeddings
    """
    matrices = []
    for model in models:
        matrices.append(model.wv.vectors)
    matrix_merged = np.mean(np.array(matrices), axis=0)
    res = models[0]
    res.wv.add_vectors(range(res.wv.vectors.shape[0]), matrix_merged, replace=True)
    return res


def load_models(party, iter):
    all_models = []
    for i in range(iter):
        all_models.append(Word2Vec.load(f"res_da/w2v_models/emb_{party}_{i}.model"))
    return all_models


def format_output(all_hits):
    res = []
    attr = []
    for hit in all_hits:
        v = []
        e = []
        a = []
        for volk in hit[0]:
            v.append(volk[0])
            a.append(volk[1])
        for elite in hit[1]:
            e.append(elite[0])
            e.append("|")
            a.append(elite[1])

        res.append((tuple(v), tuple(e)))
        attr.append(tuple(a))
    return res


def count_attr(all_hits):
    attr = []
    for hit in all_hits:
        a = []
        for volk in hit[0]:
            a.append(volk[1])
        for elite in hit[1]:
            a.append(elite[1])

        attr.append(a)
    res = list(flatten(attr))
    counter_attr.update(res)
    return res


def load_meta():
    df = pd.read_json("data/plenar_meta.json", orient="index")
    df["date"] = df.datum
    df["date"] = pd.to_datetime(df["date"], unit="ms", errors="ignore")
    return df


def print_doc(label):
    meta = load_meta()
    print(meta.loc[f'{label}.txt'])
    print(gendocs(label))


# def compute_score_sum(d):
    # return d["volk"] + d["elite"]


# def load_meta():
#     df = pd.read_json("data/plenar_meta.json", orient="index")
#     return df

if __name__ == "__main__":
    # %%
    text = "Die Deutschen finden Merkel ist nicht schlau"
    res = []
    res.append(analysis(text))

    # %%
    import spacy
    from spacy import displacy

    doc = nlp(text)
    displacy.render(doc, style="dep")

    # %%
    df = pd.DataFrame(res)
    df
    # df[df['pop'] == True]
    # %%
