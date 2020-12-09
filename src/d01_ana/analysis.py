import copy
import spacy
import pickle
import json
import random
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim import utils
from germalemma import GermaLemma
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans
from spacy.pipeline import EntityRuler
from spacy import displacy
from spacy.tokens import DocBin
from collections import Counter
from src.d00_utils.helper import chunks, flatten, filter_spans_overlap, filter_spans_overlap_no_merge
from spacy_sentiws import spaCySentiWS
from math import fabs, log
from tqdm import tqdm
from gensim.models import tfidfmodel
from pathlib import Path
from pprint import pprint
from transformers import pipeline


class SentimentRecognizer(object):

    name = "sentiment_recognizer"

    def __init__(self, nlp):
        self.load_dicts()
        # Token.set_extension('is_neg', default=False, force=True)
        # Token.set_extension('is_pos', default=False, force=True)
        Token.set_extension("is_neg", getter=self.is_neg_getter, force=True)
        Token.set_extension("is_pos", getter=self.is_pos_getter, force=True)
        Token.set_extension("is_negated", getter=self.is_negated_getter, force=True)
        Token.set_extension("span_sent", default=None, force=True)
        Doc.set_extension("has_neg", getter=self.has_neg, force=True)
        Doc.set_extension("has_pos", getter=self.has_pos, force=True)
        Span.set_extension("has_neg", getter=self.has_neg, force=True)
        Span.set_extension("has_pos", getter=self.has_pos, force=True)

    def __call__(self, doc):
        return doc

    def is_neg_getter(self, token):
        if token._.lemma in self.negativ:
            if token._.is_negated:
                return False
            else:
                return True
        if token._.lemma in self.positiv:
            if token._.is_negated:
                return True
            else:
                return False

    def is_pos_getter(self, token):
        if token._.lemma in self.positiv:
            if token._.is_negated:
                return False
            else:
                return True
        if token._.lemma in self.negativ:
            if token._.is_negated:
                return True
            else:
                return False

    def is_negated_getter(self, token):

        check = list(token.children)
        node = token
        while node.head:
            seen = node
            if seen == node.head:
                break
            check.append(node)
            check.extend(list(node.children))
            if node.head.dep_ == "pd" or node.head.dep_ == "root" or node.head.dep_ == 'rc' or node.head.dep_ == 'oc':
                check.append(node.head)
                break
            else:
                node = node.head
        attr = [
            # child for child in check if child.dep_ == "ng" or child._.lemma in negation_words
            child for child in check if child.dep_ == "ng" or child._.is_negation
        ]
        if attr:
            return True
        else:
            return False

    def load_dicts(self):
        dict_folder = "dict"
        sent = pd.read_csv(f"{dict_folder}/SentDict.csv")
        self.positiv = set([
                x.strip()
                for x in sent.loc[sent.sentiment == 1, ["feature"]]["feature"].tolist()
        ])
        self.negativ = set([
                x.strip()
                for x in sent.loc[sent.sentiment == -1, ["feature"]]["feature"].tolist()
        ])

    def has_neg(self, tokens):
        return any([t._.get("is_neg") for t in tokens])

    def has_pos(self, tokens):
        return any([t._.get("is_pos") for t in tokens])


class EntityRecognizer(object):

    name = "entity_recognizer"

    def __init__(self, nlp):
        self.load_dicts()
        self.ruler = EntityRuler(nlp, overwrite_ents=True, phrase_matcher_attr="LOWER")
        self.vocab = nlp.vocab
        patterns = []
        for term in self.dict_people:
            patterns.append({"label": "PEOPLE", "pattern": [{"_": {"lemma": term}}]})
        for term in self.dict_elite:
            patterns.append({"label": "ELITE", "pattern": [{"_": {"lemma": term}}]})
        for term in self.dict_elite_standalone:
            patterns.append(
                {"label": "ELITE_STANDALONE", "pattern": [{"_": {"lemma": term}}]}
            )
        for term in self.dict_people_ord:
            patterns.append(
                {"label": "PEOPLE_ORD", "pattern": [{"_": {"lemma": term}}]}
            )
        for term in self.dict_people_ger:
            patterns.append(
                {"label": "PEOPLE_GER", "pattern": [{"_": {"lemma": term}}]}
            )
        for term in self.dict_attr_ord:
            patterns.append({"label": "ATTR_ORD", "pattern": [{"_": {"lemma": term}}]})
        for term in self.dict_attr_ger:
            patterns.append({"label": "ATTR_GER", "pattern": [{"_": {"lemma": term}}]})
        self.ruler.add_patterns(patterns)
        # self.ruler.add_patterns([{'label': 'ELITE', 'pattern': 'europäische union'}])

        Token.set_extension("is_volk", default=False, force=True)
        Token.set_extension("is_elite", default=False, force=True)
        Token.set_extension("is_elite_neg", default=False, force=True)
        Token.set_extension("is_attr", default=False, force=True)
        Token.set_extension("attr_of", default=None, force=True)
        Doc.set_extension("has_volk", getter=self.has_volk, force=True)
        Doc.set_extension("has_elite", getter=self.has_elite, force=True)
        Span.set_extension("has_volk", getter=self.has_volk, force=True)
        Span.set_extension("has_elite", getter=self.has_elite, force=True)

    def __call__(self, doc):

        matches = self.ruler.matcher(doc)
        # matches.extend(self.ruler.phrase_matcher(doc))
        spans = []
        for id, start, end in matches:
            entity = Span(doc, start, end, label=self.vocab.strings[id])
            spans.append(entity)
        filtered = filter_spans(spans)
        for entity in filtered:
            # People setter
            if entity.label_ == "PEOPLE":
                for token in entity:
                    token._.set("is_volk", True)
            if entity.label_ == "PEOPLE_ORD":
                for token in entity:
                    check = list(token.children)
                    attr = set(
                        [
                            child
                            for child in check
                            if child._.lemma.lower() in self.dict_attr_ord
                        ]
                    )
                    if attr:
                        token._.set("is_volk", True)
                        for child in attr:
                            child._.set("is_volk", True)
                            child._.set("is_attr", True)
                            child._.set("attr_of", token.idx)

            if entity.label_ == "PEOPLE_GER" or entity.label_ == "PEOPLE_ORD":
                for token in entity:
                    check = list(token.children)
                    attr = set(
                        [
                            child
                            for child in check
                            if child._.lemma.lower() in self.dict_attr_ger
                        ]
                    )
                    if attr:
                        token._.set("is_volk", True)
                        for child in attr:
                            child._.set("is_volk", True)
                            child._.set("is_attr", True)
                            child._.set("attr_of", token.idx)
            # Elite setter
            if entity.label_ == "ELITE":
                for token in entity:
                    token._.set("is_elite", True)

                    check = list(token.children)
                    node = token
                    while node.head:
                        seen = node
                        for t in node.children:
                            if t.dep_ == "conj":
                                break
                            check.append(t)
                            # for tok in t.children:
                            # #     check.append(tok)
                            #     if tok.dep_ == "pd":
                            #         check.append(tok)
                            #     elif tok.dep_ == "mo":
                            #         check.append(tok)
                            #     elif tok.dep_ == "oa":
                            #         check.append(tok)
                            #     elif tok.dep_ == "oa2":
                            #         check.append(tok)
                            #     elif tok.dep_ == "og":
                            #         check.append(tok)
                            #     elif tok.dep_ == "da":
                            #         check.append(tok)
                            #     elif tok.dep_ == "op":
                            #         check.append(tok)
                            #     elif tok.dep_ == "cc":
                            #         check.append(tok)
                            #     elif tok.dep_ == 'avc':
                            #         check.append(tok)
                            #     elif tok.dep_ == 'app':
                            #         check.append(tok)
                            #     elif tok.dep_ == 'adc':
                            #         check.append(tok)
                            #     elif tok.dep_ == 'ag':
                            #         check.append(tok)
                        check.append(node)
                        # print(check)
                        # check.extend(list(node.children))
                        if node.head.dep_ == "pd" or node.head.dep_ == "root" or node.head.dep_ == 'rc' or node.head.dep_ == 'oc':
                            check.append(node.head)
                            break
                        # if node.head.pos_ == 'CCONJ' and node.head.text in negation_cconj:
                        if node.head.pos_ == 'CCONJ' and node.head._.is_sentence_break:
                            check.append(node.head)
                            break
                        if seen == node.head:
                            break
                        else:
                            node = node.head
                    attr = set([child for child in check if child._.is_neg])
                    if attr:
                        token._.set("is_elite_neg", True)
                        for child in attr:
                            child._.set("is_elite_neg", True)
                            child._.set("is_attr", True)
                            child._.set("attr_of", token.idx)

            # if entity.label_ == "ELITE" or entity.label_ == "ELITE_STANDALONE":
            if entity.label_ == "ELITE_STANDALONE":
                for token in entity:
                    token._.set("is_elite", True)
                    if not token._.is_negated:
                        token._.set("is_elite_neg", True)
            doc.ents = list(doc.ents) + [entity]
        # nach content analyse?
        # for span in filtered:
        # span.merge()
        return doc

    def load_dicts(self):
        dict_folder = "dict"
        # import all dicts
        # elite
        df_elite = pd.read_csv(f"{dict_folder}/elite_dict.csv")
        self.dict_elite = set(
            df_elite[df_elite.type != "elite_noneg"]["feature"].tolist()
        )
        self.dict_elite_standalone = set(
            df_elite[df_elite.type == "elite_noneg"]["feature"].tolist()
        )

        # people
        df_people = pd.read_csv(f"{dict_folder}/people_dict.csv")
        self.dict_people = set(
            df_people[df_people.type == "people"]["feature"].tolist()
        )
        self.dict_people_ord = set(
            df_people[df_people.type == "people_ordinary"]["feature"].tolist()
        )
        self.dict_attr_ord = set(
            df_people[df_people.type == "attr_ordinary"]["feature"].tolist()
        )
        self.dict_people_ger = set(
            df_people[df_people.type == "people_ger"]["feature"].tolist()
        )
        self.dict_attr_ger = set(
            df_people[df_people.type == "attr_ger"]["feature"].tolist()
        )

        # testing:
        # self.dict_people.add("wir sind das volk")
        # self.dict_elite.add("europäische union")


    # getters
    def has_volk(self, tokens):
        return any([t._.get("is_volk") for t in tokens])

    def has_elite(self, tokens):
        return any([t._.get("is_elite") for t in tokens])


class ContentAnalysis(object):
    "Runs Content Analysis as spacy-pipeline-component"
    name = "content_analysis"

    def __init__(self, nlp, window_size=25):
        self.nlp = nlp
        self.dictionary = pickle.load(open("plenar_dict.pkl", "rb"))
        # self.dictionary = None
        # Results()
        # self.res = []
        self.results = Results()
        self.window_size = window_size

        Span.set_extension(
            "has_elite_neg", getter=self.has_elite_neg_getter, force=True
        )
        Span.set_extension(
            "has_volk", getter=self.has_volk_getter, force=True
        )

    def __call__(self, doc):
        res = {
            "viz": [],
            "volk": [],
            "volk_attr": [],
            "elite": [],
            "elite_attr": [],
        }

        ##########################################
        window_size = self.window_size
        # idf_weight = 1.0
        ##########################################

        matcher = Matcher(self.nlp.vocab)
        pattern = [{"_": {"is_elite_neg": True}}]
        matcher.add("text", None, pattern)
        matches = matcher(doc)
        doclen = len(doc)

        # spans = set()
        spans = []
        token_ids = set()
        ps_counter = 1
        last_start = None
        for id, start, end in matches:
            if start - window_size < 0:
                start = 0
            else:
                start = start - window_size
            if end + window_size > doclen:
                end = doclen
            else:
                end = end + window_size
            sentence_start = doc[start].sent.start
            sentence_end = doc[end-1].sent.end
            # span = doc[start - window_size : end + window_size]
            span = {'span_start': sentence_start, 'span_end': sentence_end}
            # print(span)
            spans.append(span)

            """keep
            span = doc[sentence_start : sentence_end]
            spans.add(span)
            """

        # CAREFUL!!!!!
        spans = filter_spans_overlap_no_merge(spans)
        # print(spans)
        for span in spans:
            span = doc[span['span_start'] : span['span_end']]
            # print(span)
            if span._.has_elite_neg and span._.has_volk:
                # check sentiment of span mit sentiws
                span_sentiment = sum([token._.sentiws for token in span if token._.sentiws])
                # if span_sentiment > 0.0:
                #     pass
                    # print(span_sentiment)
                    # print(span.text)
                # else:
                for token in span:
                    token._.span_sent = span_sentiment
                    if token._.is_volk:
                        # res["viz"].append(ContentAnalysis.get_viz(token, doclen, "V", idf_weight, dictionary=self.dictionary))
                        if token._.is_attr and token.i not in token_ids:
                            res["volk_attr"].append(token._.lemma)
                            res['viz'].append(self.on_hit(token, 'VA', doc[span.start], doc[span.end-1]))
                            token_ids.add((token.i, "VA"))
                        else:
                            if token.i not in token_ids:
                                res["volk"].append(token._.lemma)
                                res['viz'].append(self.on_hit(token, 'V', doc[span.start], doc[span.end-1]))
                                token_ids.add((token.i, "V"))

                    if token._.is_elite_neg:
                        # res["viz"].append(ContentAnalysis.get_viz(token, doclen, "E", idf_weight, dictionary=self.dictionary))
                        if token._.is_attr and token.i not in token_ids:
                            res["elite_attr"].append(token._.lemma)
                            res['viz'].append(self.on_hit(token, 'EA', doc[span.start], doc[span.end-1]))
                            token_ids.add((token.i, "EA"))
                        else:
                            if token.i not in token_ids:
                                res["elite"].append(token._.lemma)
                                res['viz'].append(self.on_hit(token, 'E', doc[span.start], doc[span.end-1]))
                                token_ids.add((token.i, "E"))

        # sorts by start AND deletes duplicates!
        res["viz"] = sorted(
            [dict(t) for t in {tuple(d.items()) for d in res["viz"]}],
            key=lambda i: i["start"],
        )
        # res["c_elite"] = Counter(res["elite"])
        # self.res["token_ids"] = token_ids
        # res['doclen'] = doclen
        self.results.doclens.append(doclen)
        self.results.viz.append(res['viz'])
        # self.res.append(res)
        return doc

    # getters
    def has_elite_neg_getter(self, tokens):
        return any([t._.get("is_elite_neg") for t in tokens])

    def has_volk_getter(self, tokens):
        return any([t._.get("is_volk") for t in tokens])

    def on_hit(self, token, label, span_start, span_end):
        start = token.idx
        # end = token.idx + len(token.text) + 1
        end = token.idx + len(token.text)
        span_start_idx = span_start.idx
        span_end_idx = span_end.idx + len(span_end.text)
        lemma = token._.lemma
        score = 0.0
        # label = f"{label} | {score:.2f}"
        res = {"start": start, "end": end, "coding": label, "score": score, "lemma": lemma, "pos": token.pos_, "dep" : token.dep_, "index": token.i, "ent": token.ent_type_, "sent": token._.sentiws, "idf": self.get_idf(lemma), 'negated': token._.is_negated, "attr_of": token._.attr_of, 'isE': token._.is_elite, 'isEN': token._.is_elite_neg, 'span_start' : span_start_idx, 'span_end' : span_end_idx, 'span_sent': token._.span_sent, 'text': token.text}
        return res

    def get_idf(self, term, idf_weight=1.0):
        df = self.dictionary.dfs[self.dictionary.token2id[term.lower()]]
        return tfidfmodel.df2idf(df, self.dictionary.num_docs, log_base=2.0, add=1.0) ** idf_weight

    @staticmethod
    def get_viz(token, doclen, label, idf_weight, dictionary=None):
        start = token.idx
        end = token.idx + len(token.text) + 1
        # token = token._.lemma
        if dictionary:
            score = ContentAnalysis.compute_score_per_term(token, doclen, idf_weight, dictionary)
        else:
            score = 0.0
        label = f"{label} | {score:.2f}"
        return {"start": start, "end": end, "coding": label, "lemma": token._.lemma, 'pos': token._.pos_, 'dep' : token._.dep_}


    @staticmethod
    def viz(text, row):
        """visualize documents with displacy"""
        if isinstance(row, pd.DataFrame):
            display(row)
            viz = row.viz[0]
            ex = [
                {
                    "text": text,
                    "ents": viz,
                    "title": f"{row['doc'][0]} | {row.name_res[0]} ({row['party'][0]}) | {row['date'][0].strftime('%d/%m/%Y')}",
                    # "title": "test",
                }
            ]

        else:
            ex = [
                {
                    "text": text,
                    "ents": viz,
                    "title": "TEXT",
                }
            ]

        # find unique labels for coloring options
        all_ents = {i["label"] for i in viz}
        options = {"ents": all_ents, "colors": dict()}
        for ent in all_ents:
            if ent.startswith("E"):
                options["colors"][ent] = "coral"
            if ent.startswith("V"):
                options["colors"][ent] = "lightgrey"
            if ent.startswith("P"):
                options["colors"][ent] = "yellow"

        displacy.render(ex, style="ent", manual=True, jupyter=True, options=options)


    @staticmethod
    def compute_score_per_term(term, doclen, idf_weight, dictionary):
        score = ContentAnalysis.compute_idf(term, idf_weight, dictionary)
        ################################
        res = score / log(doclen+10, 10)
        ################################
        return res


    @staticmethod
    def compute_idf(term, idf_weight=1.0, dictionary=None):
        df = dictionary.dfs[dictionary.token2id[term.lower()]]
        return tfidfmodel.df2idf(df, dictionary.num_docs, log_base=2.0, add=1.0) ** idf_weight


    @staticmethod
    def compute_score_from_counts(counts, doclen, idf_weight, dictionary):
        scores = []
        for term, n in counts.items():
            score = ContentAnalysis.compute_score_per_term(term, doclen, idf_weight, dictionary)
            scores.append(score * n)
        res = sum(scores)
        return res


    @staticmethod
    def recount_viz(viz, doclen, dictionary, idf_weight):
        for i in viz:
            score = compute_idf(i['lemma'], idf_weight, dictionary)
            label = i['label']
            i['label'] = label.replace(label.split('| ')[1], f'{score:.2f}')
        return viz


class Results:
    """Saves relevant reslts of content analysis and contains mehtods for analysis & visualization"""
    def __init__(self):
        self.vocab = dict()
        # id2token = {value : key for (key, value) in a_dictionary.items()}
        self.labels = []
        self.viz = []
        self.doclens = []
        self.scores = []
        self.counts = []
        self.entities = set()
        self.meta_mdb = None
        self.meta_plenar = None
        self.df = None
        self.spans = None

    def __repr__(self):
        # return 'Results of Content Analysis'
        return '<{0}.{1} object at {2}>'.format(
        self.__module__, type(self).__name__, hex(id(self)))

    def __len__(self):
        return len(self.viz)

    def set_entities(self):
        for doc in self.viz:
            for hit in doc:
                if hit['ent'] == '':
                    hit['ent'] = 'ATTR'
                self.entities.add(hit['ent'])

    def load_meta():
        # self.meta_mdb = pd.read_csv('data/mdbs_metadata.csv')
        self.meta_plenar = pd.read_json('data/plenar_meta.json', orient='index')

    def compute_score(self, idf_weight=2.0, sentiment_weight=1.0, doclen_log = 10, doclen_min=100, by_doclen=True, post=False):
        scores = []
        labels = ['E', 'EA', 'V', 'VA']
        counts = []
        # seen = set()
        for i, doc in enumerate(self.viz):
            seen = set()
            score_dict = {'score': 0.0}
            count_dict = {}
            for ent in self.entities:
                score_dict[ent] = 0.0
            for label in labels:
                score_dict[label] = 0.0
                count_dict[label] = Counter()
            for hit in doc:
                if not hit['sent']:
                    hit['sent'] = 0.0
                if post:
                    # print('is POST')
                    if hit['TOK_IS_POP'] and hit['SPAN_IS_POP'] and hit['start'] not in seen:
                        # score = (hit['idf'] ** idf_weight) * ((1+fabs(hit['sent'])) ** sentiment_weight) / log(self.doclens[i]+doclen_weight, 10)
                        score = (hit['idf'] ** idf_weight) * ((1+fabs(hit['sent'])) ** sentiment_weight)
                        seen.add(hit['start'])
                    else:
                        score = 0.0
                else:
                    if hit['start'] not in seen:
                        # score = (hit['idf'] ** idf_weight) * ((1+fabs(hit['sent'])) ** sentiment_weight) / log(self.doclens[i]+doclen_weight, 10)
                        score = (hit['idf'] ** idf_weight) * ((1+fabs(hit['sent'])) ** sentiment_weight)
                        seen.add(hit['start'])
                if by_doclen:
                    score = score / log(self.doclens[i] + doclen_min, doclen_log)
                hit['score'] = score
                score_dict['score'] += score
                score_dict[hit['ent']] += score
                score_dict[hit['coding']] += score
                count_dict[hit['coding']].update([hit['lemma']])
            # for label in labels:
            #     count_dict[label] = Counter(count_dict[label])
            counts.append(count_dict)
            scores.append(score_dict)
        self.scores = scores
        self.counts = counts


    def compute_score_spans(self):
        span_dict = {}
        # {doc: [(span_start, span_end, score_sum)]}
        for i, doc in enumerate(self.viz):
            label = self.labels[i]
            span_dict[label] = {}
            # scores = []
            for hit in doc:
                span_start = hit['span_start']
                span_end = hit['span_end']
                span_id = (span_start, span_end)
                if span_id not in span_dict[label]:
                    span_dict[label][span_id] = 0.0
                span_dict[label][span_id] += hit['score']

            # span_dict[label] = sorted(
            #     [dict(t) for t in {tuple(d.items()) for d in res["viz"]}],
            #     key=lambda i: i["start"],
            # )

        self.spans = span_dict


    def top_spans(self, topn=10):
        all_spans = []
        for doc in self.spans.items():
            for span in doc[1]:
                all_spans.append((doc[0], span, self.spans[doc[0]][span]))
        all_spans.sort(key = lambda tup: tup[2], reverse=True)
        return all_spans[:topn]


    def create_df(self):
        # df = pd.DataFrame.from_dict({'doc': self.labels}, {'doclen': self.doclens}, {'scores': self.scores})
        df = pd.DataFrame.from_dict({'doc': self.labels, 'doclen': self.doclens, 'scores': self.scores})
        df = pd.concat([df.drop('scores', axis=1), df.scores.apply(pd.Series)], axis=1, sort=False)
        self.df = df


    def prepare(self, post=False):
        self.set_entities()
        self.compute_score(by_doclen=True, idf_weight=1.5, doclen_log=10, post=post)
        self.compute_score_spans()
        self.create_df()
        self.add_meta_plenar()


    def visualize(self, label, span=None, filter_by=False, pres=False):
        row = self.df.loc[self.df['doc'] == label]
        text = gendocs(label)
        viz = copy.deepcopy(self.viz[self.labels.index(label)])
        # pprint(viz)
        Results.render(text, row, viz, span=span, filter_by=filter_by, pres=pres)


    @staticmethod
    def render(text, row, viz, span=None, filter_by=['score'], pres=False):
        """visualize documents with displacy"""

        def filter_by_condition(viz, condition):
            viz = [i for i in viz if i[condition]]
            return viz

        viz = Results.filter_viz(viz, on='start')
        viz = filter_spans_overlap(viz)
        viz_span = []

        if span:
            span = span
        else:
            span = (0, len(text) + 1)

        if pres:
            viz_span_ = []
            for hit in viz:
                paragraph = {}
                hit['start'] -= span[0]
                hit['end'] -= span[0]
                paragraph['start'] = hit['span_start']
                paragraph['end'] = hit['span_end']
                # hit['label'] = f"{hit['coding']} | {hit['score']:.2f}"
                if paragraph['start'] not in [i['start'] for i in viz_span_]:
                    viz_span_.append(paragraph)

            for n, v in enumerate(viz_span_):
                viz_span.append({'start': v['start'], 'end': v['end'], 'label': f'P|{n+1}'})

            viz_span = sorted(viz_span, key=lambda x: x['start'])

        ##################################################
        else:

            if filter_by:
                for condition in filter_by:
                    viz = filter_by_condition(viz, condition)

            if span[0] > 0:
                viz = [i for i in viz if i['span_start'] == span[0]]

            for hit in viz:

                hit['start'] -= span[0]
                hit['end'] -= span[0]

                hit['label'] = f"{hit['coding']} | {hit['score']:.2f}"
                viz_span.append(hit)

            viz_starts = set([i['span_start'] for i in viz])

            for n, start in enumerate(sorted(viz_starts)):
                if start > 0 and span[0] == 0:
                    viz_span.append({'start': start-1, 'end': start, 'label': f'P{n+1} | {start}'})

            viz_span = sorted(viz_span, key=lambda x: x['start'])
        ###############################################

        ex = [
            {
                "text": text[span[0]: span[1]],
                "ents": viz_span,
                "title": f"{row['doc'][0]} | {row.name_res[0]} ({row['party'][0]}) | {row['date'][0].strftime('%d/%m/%Y')}",
                # "title": f"{row['doc']} | {row.name_res} ({row['party']}) | {row['date'].strftime('%d/%m/%Y')}",
                # 'title': 'text'
            }
        ]
        all_ents = {i["label"] for i in viz_span}

        # else:
        #     viz_all = []
        #     for hit in viz:
        #         hit['label'] = f"{hit['coding']} | {hit['score']:.2f}"
        #         viz_all.append(hit)
        #     ex = [
        #         {
        #             "text": text,
        #             "ents": viz_all,
        #             "title": f"{row['doc'][0]} | {row.name_res[0]} ({row['party'][0]}) | {row['date'][0].strftime('%d/%m/%Y')}",
        #         }
        #     ]
        #     # find unique labels for coloring options
        #     all_ents = {i["label"] for i in viz_all}

        options = {"ents": all_ents, "colors": dict()}
        for ent in all_ents:
            if ent.startswith("E"):
                options["colors"][ent] = "coral"
            if ent.startswith("V"):
                options["colors"][ent] = "lightgrey"
            if ent.startswith("P"):
                options["colors"][ent] = "yellow"

        displacy.render(ex, style="ent", manual=True, jupyter=True, options=options)

    def add_meta_plenar(self):
        df = pd.read_json("data/plenar_meta.json", orient="index")
        # keep for future
        # dfval_2 = pd.read_json('/media/philippy/SSD/data/ma/corpus/presse_meta.json', orient='index')
        # dfval_3 = pd.read_json('/media/philippy/SSD/data/ma/corpus/twitter_meta.json', orient='index')
        # dfval = dfval_1.append([dfval_2, dfval_3])
        df["doc"] = df.index
        df["doc"] = df.doc.apply(lambda x: x.split(".")[0])
        # fix timestamps
        df["date"] = df.datum
        df["date"] = pd.to_datetime(df["date"], unit="ms", errors="ignore")
        # merge results and meta
        dfs = self.df.merge(df.loc[:, ["date", "party", "doc", "name_res", "gender", "election_list", "education", "birth_year"]], how="left", on="doc")
        dfs = dfs.set_index("date").loc["2013-10-01":"2020-01-01"]
        dfs["date"] = dfs.index
        self.df = dfs

    def evaluate_by_category(self, category, target):
        grouped = self.df.groupby(category).mean().sort_values(target, ascending=False)
        # mdbs_meta = pd.read_csv('data/mdbs_metadata.csv')
        # res = pd.merge(grouped, mdbs_meta, how='left', on=category)
        return grouped

    def top_terms(self, cat=False, abs=True, party=None, topn=100):
        if party:
            df = self.df.loc[self.df.party == party].copy()
        else:
            df = self.df.copy()
        if abs:
            labels = [i for i in df.doc]
            ids = []
            for label in labels:
                ids.append(self.labels.index(label))
            res = []
            for i, count in enumerate(self.counts):
                if i in ids:
                    if cat:
                        res.append(count[cat])
                    else:
                        res.extend(count.values())
            # res = df.apply(lambda row: Counter(row.counts[cat]), axis=1)
            res = sum([i for i in res], Counter())
        else:
            labels = [i for i in df.doc]
            ids = []
            for label in labels:
                ids.append(self.labels.index(label))

            score_dict = {}

            for i, doc in enumerate(self.viz):
                if i in ids:
                    for hit in doc:
                        if cat:
                            if hit['coding'] == cat:
                                if hit['lemma'] not in score_dict:
                                    score_dict[hit['lemma']] = 0.0
                                score_dict[hit['lemma']] += hit['score']
                        else:
                            if hit['lemma'] not in score_dict:
                                score_dict[hit['lemma']] = 0.0
                            score_dict[hit['lemma']] += hit['score']
            res = score_dict

        return dict(sorted(res.items(), key=lambda x: x[1], reverse=True)[:topn])


    def coded(self, label, index_start, categories=None):
        for hit in self.viz[self.labels.index(label)]:
            # if hit['lemma'] == 'steuerzahler':
            if hit['span_start'] == index_start:
                if not categories:
                    hits.apend(hit)
                else:
                    return({cat: hit[cat] for cat in categories})
        return(hits)


    def coding(self):
        res_viz = []
        for i, (doc, doc_vizs) in enumerate(zip(self.spans, self.viz)):
            # if i % 500 == 0:
            #     print(i, f'/{len(self.spans)}')
            doc_viz = []
            # doc_vizs = Results.filter_viz(doc_vizs, on='start')
            for span in self.spans[doc]:
                viz = []
                text = gendocs(doc)[span[0]:span[1]]
                viz.extend([viz for viz in doc_vizs if viz['span_start'] == span[0] and viz['span_end'] == span[1]])

                # final coding
                pop_hits_v = 0
                pop_hits_e = 0
                for v in viz:
                    v['TOK_IS_POP'] = False
                    v['SPAN_IS_POP'] = False

                    if v['RLY_GER'] and (v['RLY_V'] == True or v['RLY_E'] == True):
                        v['TOK_IS_POP'] = True
                    if v['TOK_IS_POP'] and v['coding'] == 'V':
                        pop_hits_v += 1
                        for attr in viz:
                            if attr['attr_of'] == v['start']:
                                attr['RLY_V'] = True
                                attr['TOK_IS_POP'] = True
                    if v['TOK_IS_POP'] and (v['coding'] == 'E' or (v['coding'] == 'EA' and v['pos'] == 'NOUN')):
                        pop_hits_e += 1
                        for attr in viz:
                            if attr['attr_of'] == v['start']:
                                attr['RLY_E'] = True
                                attr['TOK_IS_POP'] = True

                if pop_hits_v > 0 and pop_hits_e > 0:
                    for v in viz:
                        v['SPAN_IS_POP'] = True
                doc_viz.extend(viz)
            res_viz.append(doc_viz)
        self.viz = res_viz


    def coding_pop(self, idf_weight=1.5, sentiment_weight=1.0):
        self.set_entities()
        self.coding()
        self.compute_score(by_doclen=True, idf_weight=idf_weight, sentiment_weight=sentiment_weight, doclen_log=2, post=True)
        self.create_df()
        self.add_meta_plenar()


    def filter_res(self, label):
        res = Results()
        id = self.labels.index(label)
        res.viz = [self.viz[id]]
        res.labels = [self.labels[id]]
        res.doclens = [self.doclens[id]]
        res.scores = [self.scores[id]]
        res.spans = {label: self.spans[label]}
        return res


    @staticmethod
    def filter_viz(viz, on='start'):
        res = []
        ids = set()
        for hit in viz:
            if hit[on] not in ids:
                res.append(hit)
                ids.add(hit[on])

        return res

    @staticmethod
    def visualize_text(text, row):
        """visualize documents with displacy"""
        if isinstance(row, pd.DataFrame):
            display(row)
            viz = row.viz[0]
            ex = [
                {
                    "text": text,
                    "ents": viz,
                    "title": f"{row['doc'][0]} | {row.name_res[0]} ({row['party'][0]}) | {row['date'][0].strftime('%d/%m/%Y')}",
                    # "title": "test",
                }
            ]

        else:
            viz = row
            for hit in viz:
                hit['label'] = f"{hit['label']} | {hit['score']:.2f}"
            ex = [
                {
                    "text": text,
                    "ents": viz,
                    "title": "TEXT",
                }
            ]

        # find unique labels for coloring options
        all_ents = {i["label"] for i in viz}
        options = {"ents": all_ents, "colors": dict()}
        for ent in all_ents:
            if ent.startswith("E"):
                options["colors"][ent] = "coral"
            if ent.startswith("V"):
                options["colors"][ent] = "lightgrey"
            if ent.startswith("P"):
                options["colors"][ent] = "yellow"

        displacy.render(ex, style="ent", manual=True, jupyter=True, options=options)


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


def custom_extensions(doc):

    lemmatizer = GermaLemma()
    negation_words = set(["nie", "keinsterweise", "keinerweise", "niemals", "nichts", "kaum", "keinesfalls", "ebensowenig", "nicht", "kein", "keine", "weder"])
    negation_cconj = set(['aber', 'jedoch', 'doch', 'sondern'])

    def lemma_getter(token):
        # if " " in token.text:
        #     return token.lemma_.lower()
        try:
            return lemmatizer.find_lemma(token.text, token.tag_).lower()
        except:
            return token.lemma_.lower()

    def is_negation_getter(token):
        if token._.lemma in negation_words:
            return True
        else:
            return False

    def is_sentence_break_getter(token):
        if token._.lemma in negation_cconj:
            return True
        else:
            return False

    Token.set_extension("lemma", getter=lemma_getter, force=True)
    Token.set_extension("is_negation", getter=is_negation_getter, force=True)
    Token.set_extension("is_sentence_break", getter=is_sentence_break_getter, force=True)
    return doc


def recount_viz(df, dictionary, idf_weight):
    df['viz'] = df.apply(lambda row: ContentAnalysis.recount_viz(row['viz'], row['doclen'], dictionary, idf_weight), axis=1)


def compute_score_from_df(df, dictionary, idf_weight=1.0):
    cols = ['viz', 'volk', 'volk_attr', 'elite', 'elite_attr']
    for col in cols:
        df[col] = df.apply(lambda row: eval(str(row[col])), axis=1)
    for col in cols[1:]:
        df[f'c_{col}'] = df.apply(lambda row: Counter(row[col]), axis=1)
        df[f'score_{col}'] = df.apply(lambda row: ContentAnalysis.compute_score_from_counts(row[f'c_{col}'], row['doclen'], idf_weight, dictionary), axis=1)
    df['score'] = df.apply(lambda row: sum([row[f'score_{col}'] for col in cols[1:]]), axis=1)


# def evaluate_by_category(category, target, df):
#     grouped = df.groupby(category).mean().sort_values(target, ascending=False)
#     mdbs_meta = pd.read_csv('data/mdbs_metadata.csv')
#     res = pd.merge(grouped, mdbs_meta, how='left', on=category)
#     display(res)


def serialize(directory, party='all', sample=None):

    Path(f"nlp/{directory}/docs").mkdir(parents=True, exist_ok=False)

    doc_labels = load_data(party)
    if type(sample) == int:
        doc_labels = random.sample(doc_labels, sample)
        text = None
    elif type(sample) == str:
        doc_labels = ['test']
        text = sample
    elif type(sample) == list:
        doc_labels = sample
        text = None
    else:
        text = None
    print("Number of documents: {}".format(len(doc_labels)))
    print(f"Beginning Serialization with parameters: \n Party: {party}")
    nlp = spacy.load("de_core_news_lg")
    ca = ContentAnalysis(nlp)
    entity_recognizer = EntityRecognizer(nlp)
    sentiment_recognizer = SentimentRecognizer(nlp)
    # nlp.add_pipe(ca, last=True)
    nlp.add_pipe(custom_lemma, last=True)
    nlp.add_pipe(sentiment_recognizer, last=True)
    nlp.add_pipe(entity_recognizer, last=True)
    nlp.remove_pipe("ner")
    labels = []
    # doc_bin = DocBin(attrs=["LEMMA", "POS", "DEP", "ENT_TYPE"], store_user_data=True)
    for label in tqdm(doc_labels):
        labels.append(label)
        if text:
            doc = nlp(text)

        else:
            doc = nlp(gendocs(label))
        # json_doc = doc.to_json(['has_volk', 'has_elite'])
        # doc_bin.add(doc)
        # with open(f'nlp/test/{label}.json', 'w') as outfile:
        #     json.dump(json_doc, outfile)
        doc_bytes = doc.to_bytes()
        with open(f'nlp/{directory}/docs/{label}', 'wb') as f:
            f.write(doc_bytes)
    # nlp.to_disk('nlp/test')
    # data = doc_bin.to_bytes()
    # with open(f'nlp/{directory}/docs_plenar', 'wb') as f:
    #     f.write(data)
    nlp.vocab.to_disk(f'nlp/{directory}/vocab.txt')
    with open(f'nlp/{directory}/labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    print(f"Serialization complete. \nResults saved in nlp/{directory}/")


def hash(w):
    return int(hashlib.md5(w.encode('utf-8')).hexdigest()[:9], 16)


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
        all_models.append(Word2Vec.load(f'res_da/w2v_models/emb_{party}_{i}.model'))
    return all_models


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


def lemma_getter(token):
    # if " " in token.text:
    #     return token.lemma_.lower()
    try:
        return lemmatizer.find_lemma(token.text, token.tag_).lower()
    except:
        return token.lemma_.lower()



