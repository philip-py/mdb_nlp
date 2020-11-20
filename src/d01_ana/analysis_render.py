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
class ContentAnalysisRender:
    def __init__(self, model, dict_folder, directory, render=False):

        self.nlp = spacy.load(model)
        self.folder = directory

        if not render:
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

    def analyze(self, label, text, window_size=30):

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
                # print(token.text)
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
                viz.append(get_viz(token, 'E'))
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
                        # print(check)
                        break

                # experimental for viz
                attr_token = set([child for child in check if child._.is_neg])
                # print(attr_token)
                for child in attr_token:
                    viz.append(get_viz(child, 'E'))
                del attr_token
                # experiment end

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
            # print(token._.lemma)

            check = list(token.children)
            # print(check)

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

        def get_viz(token, label):
            start = token.idx
            end = token.idx + len(token.text) + 1
            score = compute_per_term(token._.lemma, len(doc), 2.0)
            label = f'{label} | {score:.2f}'
            return {'start': start, 'end': end, 'label': label}

        def get_viz_start(token, label):
            start_id = token.i
            tok = doc[start_id-1]
            start = tok.idx
            end = tok.idx + len(tok.text)
            label = f'{label}'
            return {'start': start, 'end': end, 'label': label}

        # def get_viz_end(token, label):
        #     start_id = token.i
        #     tok = doc[start_id-1]
        #     start = tok.idx
        #     end = tok.idx + len(tok.text)
        #     label = f'{label}'
        #     return {'start': start, 'end': end, 'label': label}


        def compute_per_term(term, doclen, idf_weight):
            score = compute_idf(term) ** idf_weight
            res = score / log(doclen+10, 10)
            return res

        def compute_idf(term):
            df = dictionary.dfs[dictionary.token2id[term.lower()]]
            score = tfidfmodel.df2idf(df, number_docs, log_base=2.0, add=0.0)
            return score


        number_docs = 29500
        dictionary = pickle.load(open('plenar_dict.pkl', 'rb'))

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
        viz = []
        ###########################################################
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

                # print(token._.lemma)

                # viz = [{"start": 4, "end": 10, "label": f"ORG | {0.0}"}]
                if token._.is_volk:
                    hits["volk"].append(token._.lemma)
                    hits["volk_text"].append(token.text)
                    viz.append(get_viz(token, 'V'))
                    # start = token.idx
                    # end = token.idx + len(token.text) + 1
                    # viz.append({'start': start, 'end': end, 'label': 'V | 0.0'})

                if token._.is_neg_elite:
                    hits["elite"].append(token._.lemma)
                    hits["elite_text"].append(token.text)
                    hits["attr"].append(token._.info)
                    all_sents.append(sent)
                    viz.append(get_viz(token, 'E'))
                    # start = token.idx
                    # end = token.idx + len(token.text) + 1
                    # viz.append({'start': start, 'end': end, 'label': 'E | 1.0'})
                # Token.set_extension('is_pos_volk', getter=is_pos_volk_getter_func, force=True)

                # print(token.text, token.lemma_, token._.lemma, token.pos_)
                # print(list(token.children))

        matcher = Matcher(self.nlp.vocab)
        pattern = [{"_": {"is_neg_elite": True}}]
        matcher.add("text", None, pattern)
        matches = matcher(doc)
        has_pop = set()
        has_pop_ = set()
        # tokens_pop = []
        info = []
        for j, (match_id, start, end) in enumerate(matches):
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
                    # print(type(doc[sentence_start]))
                    viz.append(get_viz_start(doc[sentence_start], f'PS | {j}'))
                    # viz.append(get_viz_end(doc[sentence_end], 'PE({j})'))
                    # viz.append({'start': sentence_start, 'end': sentence_start+20, 'label': f'P'})
                    # viz.append({'start': sentence_end-10, 'end': sentence_end+1, 'label': f'P | end'})


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
        viz = [dict(t) for t in {tuple(d.items()) for d in viz}]
        res_dict["viz"] = sorted(viz, key=lambda i: i['start'])
        # res.append(res_dict)
        return res_dict


