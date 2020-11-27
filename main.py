#%%
# 2do:
# DO THE EVALUATION WITH 50 TEXTS
# FINAL CA!
# FINISH THE EXPOSE!

# problem souverän?

# %%
# import plac
import spacy
import pandas as pd
from germalemma import GermaLemma
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans
from spacy.pipeline import EntityRuler
from spacy import displacy
from spacy.tokens import DocBin
from collections import Counter
# from src.d01_ana.analysis import *
from src.d00_utils.helper import chunks, flatten
from spacy_sentiws import spaCySentiWS
from math import fabs
from tqdm import tqdm
from gensim.models import tfidfmodel
import pickle
import json
import random

negation_words = set(["nie", "keinsterweise", "keinerweise", "niemals", "nichts", "kaum", "keinesfalls", "ebensowenig", "nicht", "kein", "keine", "weder"])
negation_cconj = set(['aber', 'jedoch', 'doch', 'sondern'])
sentiws = spaCySentiWS(sentiws_path='sentiws/')


class SentimentRecognizer(object):

    name = "sentiment_recognizer"

    def __init__(self, nlp):
        self.load_dicts()
        # Token.set_extension('is_neg', default=False, force=True)
        # Token.set_extension('is_pos', default=False, force=True)
        Token.set_extension("is_neg", getter=self.is_neg_getter, force=True)
        Token.set_extension("is_pos", getter=self.is_pos_getter, force=True)
        Token.set_extension("is_negated", getter=self.is_negated_getter, force=True)
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
            child for child in check if child.dep_ == "ng" or child._.lemma in negation_words
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
                        if node.head.pos_ == 'CCONJ' and node.head.text in negation_cconj:
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

    def __init__(self, nlp):
        self.nlp = nlp
        self.dictionary = pickle.load(open("plenar_dict.pkl", "rb"))
        # self.dictionary = None
        # Results()
        # self.res = []
        self.results = Results()

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
        window_size = 25
        # idf_weight = 1.0
        ##########################################

        matcher = Matcher(self.nlp.vocab)
        pattern = [{"_": {"is_elite_neg": True}}]
        matcher.add("text", None, pattern)
        matches = matcher(doc)
        doclen = len(doc)

        spans = set()
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
            span = doc[sentence_start : sentence_end]
            spans.add(span)
        for span in spans:
            if span._.has_elite_neg and span._.has_volk:
                span_sentiment = sum([token._.sentiws for token in span if token._.sentiws])
                if span_sentiment > 0.0:
                    pass
                    # print(span_sentiment)
                    # print(span.text)
                else:
                    for token in span:
                        if token._.is_volk:
                            # res["viz"].append(ContentAnalysis.get_viz(token, doclen, "V", idf_weight, dictionary=self.dictionary))
                            if token._.is_attr and token.i not in token_ids:
                                res["volk_attr"].append(token._.lemma)
                                res['viz'].append(self.on_hit(token, 'VA'))
                                token_ids.add((token.i, "VA"))
                            else:
                                if token.i not in token_ids:
                                    res["volk"].append(token._.lemma)
                                    res['viz'].append(self.on_hit(token, 'V'))
                                    token_ids.add((token.i, "V"))

                        if token._.is_elite_neg:
                            # res["viz"].append(ContentAnalysis.get_viz(token, doclen, "E", idf_weight, dictionary=self.dictionary))
                            if token._.is_attr and token.i not in token_ids:
                                res["elite_attr"].append(token._.lemma)
                                res['viz'].append(self.on_hit(token, 'EA'))
                                token_ids.add((token.i, "EA"))
                            else:
                                if token.i not in token_ids:
                                    res["elite"].append(token._.lemma)
                                    res['viz'].append(self.on_hit(token, 'E'))
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

    def on_hit(self, token, label):
        start = token.idx
        end = token.idx + len(token.text) + 1
        lemma = token._.lemma
        score = 0.0
        # label = f"{label} | {score:.2f}"
        res = {"start": start, "end": end, "label": label, "score": score, "lemma": lemma, "pos": token.pos_, "dep" : token.dep_, "index": token.i, "ent": token.ent_type_, "sent": token._.sentiws, "idf": self.get_idf(lemma), 'negated': token._.is_negated}
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
        return {"start": start, "end": end, "label": label, "lemma": token._.lemma, 'pos': token._.pos_, 'dep' : token._.dep_}


    @staticmethod
    def get_viz_start(doc, token, label):
        start_id = token.i
        if start_id != 0:
            start_id =- 1
        tok = doc[start_id]
        start = tok.idx
        end = tok.idx + len(tok.text)
        label = f"{label}"
        return {"start": start, "end": end, "label": label}


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

    def __repr__(self):
        return 'results of Content Analysis'

    def set_entities(self):
        for doc in self.viz:
            for hit in doc:
                if hit['ent'] == '':
                    hit['ent'] = 'ATTR'
                self.entities.add(hit['ent'])

    def load_meta():
        # self.meta_mdb = pd.read_csv('data/mdbs_metadata.csv')
        self.meta_plenar = pd.read_json('data/plenar_meta.json', orient='index')

    def compute_score(self, idf_weight=2.0, sentiment_weight=1.0, doclen_weight=100):
        scores = []
        labels = ['E', 'EA', 'V', 'VA']
        counts = []
        for i, doc in enumerate(self.viz):
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
                score = (hit['idf'] ** idf_weight) * ((1+fabs(hit['sent'])) ** sentiment_weight) / log(self.doclens[i]+doclen_weight, 10)
                hit['score'] = score
                score_dict['score'] += score
                score_dict[hit['ent']] += score
                score_dict[hit['label']] += score
                count_dict[hit['label']].update([hit['lemma']])
            # for label in labels:
            #     count_dict[label] = Counter(count_dict[label])
            counts.append(count_dict)
            scores.append(score_dict)
        self.scores = scores
        self.counts = counts

    def create_df(self):
        # df = pd.DataFrame.from_dict({'doc': self.labels}, {'doclen': self.doclens}, {'scores': self.scores})
        df = pd.DataFrame.from_dict({'doc': self.labels, 'doclen': self.doclens, 'scores': self.scores})
        df = pd.concat([df.drop('scores', axis=1), df.scores.apply(pd.Series)], axis=1, sort=False)
        self.df = df

    def visualize(self, label):
        id = self.labels.index(label)
        text = gendocs(label)
        # meta = self.meta_mdb.loc[]
        Results.render(text, self.viz[id])

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

    def top_terms(self, cat=None, abs=True, party=None):
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
                            if hit['label'] == cat:
                                if hit['lemma'] not in score_dict:
                                    score_dict[hit['lemma']] = 0.0
                                score_dict[hit['lemma']] += hit['score']
                        else:
                            if hit['lemma'] not in score_dict:
                                score_dict[hit['lemma']] = 0.0
                            score_dict[hit['lemma']] += hit['score']
            res = score_dict

        return dict(sorted(res.items(), key=lambda x: x[1], reverse=True))

    @staticmethod
    def render(text, row):
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


def custom_lemma(doc):

    lemmatizer = GermaLemma()

    def lemma_getter(token):
        # if " " in token.text:
        #     return token.lemma_.lower()
        try:
            return lemmatizer.find_lemma(token.text, token.tag_).lower()
        except:
            return token.lemma_.lower()

    Token.set_extension("lemma", getter=lemma_getter, force=True)
    return doc


def custom_extensions(doc):

    negation_words = set(["nie", "keinsterweise", "keinerweise", "niemals", "nichts", "kaum", "keinesfalls", "ebensowenig", "nicht", "kein", "keine", "weder"])
    negation_cconj = set(['aber', 'jedoch', 'doch', 'sondern'])

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

    Token.set_extension("is_negation", getter=is_negation_getter, force=True)

    Token.set_extension("is_sentence_break", getter=is_sentence_break_getter), force=True)

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


def evaluate_by_category(category, target, df):
    grouped = df.groupby(category).mean().sort_values(target, ascending=False)
    mdbs_meta = pd.read_csv('data/mdbs_metadata.csv')
    res = pd.merge(grouped, mdbs_meta, how='left', on=category)
    display(res)


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


def content_analysis(directory, party="all", sample=None, debug=False):

    # if os.path.isdir(f'res_ca/{directory}'):
    #     print("Directory already exists.")
    #     return
    if directory != 'test':
        Path(f"res_ca/{directory}/").mkdir(parents=False, exist_ok=False)

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
    print(f"Beginning Content Analysis with parameters: \n Party: {party}")
    nlp = spacy.load("de_core_news_lg")
    ca = ContentAnalysis(nlp)
    entity_recognizer = EntityRecognizer(nlp)
    sentiment_recognizer = SentimentRecognizer(nlp)
    nlp.add_pipe(ca, last=True)
    nlp.add_pipe(custom_lemma, before="content_analysis")
    nlp.add_pipe(sentiment_recognizer, before="content_analysis")
    nlp.add_pipe(sentiws, before='content_analysis')
    nlp.add_pipe(entity_recognizer, before="content_analysis")
    nlp.remove_pipe("ner")
    labels = []
    for label in tqdm(doc_labels):
        labels.append(label)
        if text:
            doc = nlp(text)
            if debug:
                for token in doc:
                    print(token.text, token.ent_type_, token._.is_elite_neg, token._.is_attr, token._.is_negated)
        else:
            doc = nlp(gendocs(label))
        ca.results.labels.append(label)
    with open(f'res_ca/{directory}/labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    with open(f'res_ca/{directory}/results_all.pkl', 'wb') as f:
        pickle.dump(ca.results, f)
    print(f"Content Analysis complete. \nResults saved in {directory}/...")

    return ca.results
    # return (ca.results, doc)


# %%
if __name__ == "__main__":
    res = content_analysis('test', party='all', sample=10_000)
    # res = pickle.load(open("res_ca/1127/results_all.pkl", "rb"))
    res.set_entities()
    res.compute_score()
    res.create_df()
    res.add_meta_plenar()
    display(res.df)

# %%
res.df.groupby('party').mean()

# %%
pd.set_option('display.max_rows', 50)
Results.evaluate_by_category = evaluate_by_category
# by_name = res.evaluate_by_category('name_res', 'score')
# by_name[['name_res', 'party', 'score']].head(50)
by_state = res.evaluate_by_category('election_list', 'score')
# by_state[['election_list', 'score']].head(50)

# %%
grouped = res.df.groupby('election_list').mean().sort_values('score', ascending=False)

# %%
df.groupby('party').mean()
# df.sort_values('score', ascending=False).loc[:, ['name_res', 'party', 'score', 'doclen', 'doc']].head(100)

# %%
by_name = evaluate_by_category('name_res', 'score', df)
# by_name[:100].party.value_counts()
by_name.loc[:, ['name_res', 'party', 'doclen', 'score']]

# %%
# EVALUATE
eval = df.sort_values("score", ascending=False)[["party", "doc"]][:500]
labels = [i for i in eval.doc]
display(eval.head(50))

# %%
viz = res.df.groupby(["party"]).resample("Q").mean().reset_index()
viz.drop(
    viz[(viz["party"] == "Parteilos") | (viz["party"] == "Die blaue Partei")].index,
    inplace=True,
)
viz2 = res.df.resample("Q").mean().reset_index()
import plotly.express as px
import plotly.graph_objects as go

fig = px.line(x=viz.date, y=viz.score, color=viz.party, title="Populism-Score")
fig.update_layout(width=1000, title_font_size=20)

colors = ["darkblue", "black", "blue", "green", "magenta", "gold", "red"]
for j, party in enumerate([i.name for i in fig.data]):
    fig.data[j].line.color = colors[j]
fig.add_trace(
    go.Scatter(
        x=viz2.date,
        y=viz2.score,
        mode="lines",
        name="Durchschnit / Quartal",
        marker_symbol="pentagon",
        line_width=5,
        line_color="darkgrey",
        line_dash="dash",
    )
)
fig.layout.template = "plotly_white"
fig.layout.legend.title.text = "Partei"
fig.layout.xaxis.title.text = "Jahr"
fig.layout.yaxis.title.text = "Score"
fig.update_traces(hovertemplate="Score: %{y} <br>Jahr: %{x}")
for i in range(2, 7):
    fig.data[i].visible = "legendonly"
fig.show()

# %%
def viz_id(df, id):
    ContentAnalysis.viz(gendocs(id), df.loc[df['doc'] == id])

# for label in labels[:50]:
    # viz_id(df, label)
viz_id(df, 'plenar_014102')

# %%
pd.set_option('display.max_rows', 100)
# %%

# %%
# de-serialize | working!
Token.set_extension("is_volk", default=False)
Token.set_extension("is_elite", default=False)
Token.set_extension("is_elite_neg", default=False)
Token.set_extension("is_attr", default=False)
Token.set_extension('lemma', default=None)

nlp = spacy.blank('de')
# nlp = nlp.from_disk('nlp/test')
nlp.vocab.from_disk('nlp/1125/vocab.txt')
with open('nlp/1125/docs/plenar_000032', 'rb') as f:
    doc_bytes = f.read()


doc = spacy.tokens.Doc(nlp.vocab).from_bytes(doc_bytes)
for token in doc:
    if token._.is_volk:
        print(token._.lemma)
# for entity in doc.ents:
#     print(entity.label_)

ca = ContentAnalysis(nlp)
nlp.add_pipe(ca, last=True)
print(nlp.pipe_names)

new = nlp(doc.text)
print(ca.res)


# %%
import statsmodels.api as sm

df_reg = res.df.copy()
# change daterange?
df_reg = df_reg.loc["2013-10-01":"2020-01-01"]
# df_reg = df_reg.loc[df['pop'] == True]
reg = df_reg[["date", "score", "party", "birth_year"]].dropna()


def sommerpause(row):
    if 7 < row.month < 9:
        return "sommerpause"
    else:
        return "keine_sommerpause"


reg["year"] = reg["date"].dt.strftime("%y")
reg["year"] = reg["year"].astype("int")
reg["year"] = reg["year"] - 13
reg["month"] = reg["date"].dt.strftime("%m")
reg["month"] = reg["month"].astype("int")
reg["is_summer"] = reg.apply(lambda row: sommerpause(row), axis=1)

sm.add_constant(reg)

# regression = sm.Poisson.from_formula(
    # "score ~ year * C(party, Treatment('CDU'))", reg, missing="drop"
# ).fit()
regression = sm.Poisson.from_formula("score ~ C(party, Treatment('CDU')) + year", reg, missing="drop").fit()
# res = sm.OLS.from_formula("score ~ C(opp, Treatment('not_opp'))", reg, missing='drop').fit()
# res = sm.Poisson.from_formula("score ~ C(party, Treatment('CDU')) + date * C(party, Treatment('CDU'))", reg, missing='drop').fit()

sum = regression.summary()
sum
# %%
