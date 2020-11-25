# %%
import plac
import spacy
import pandas as pd
from germalemma import GermaLemma
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans
from spacy.pipeline import EntityRuler
from spacy import displacy
from collections import Counter
from src.d01_ana.analysis import *
from src.d00_utils.helper import chunks, flatten
import pickle

negation_words = ["nie", "keinsterweise", "keinerweise", "niemals", "nichts", "kaum", "keinesfalls", "ebensowenig", "nicht", "kein", "weder"]


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
            child for child in check if child.dep_ == "ng" or child.dep_ in negation_words
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
                        # check.extend(list(node.children))
                        if node.head.dep_ == "pd" or node.head.dep_ == "root" or node.head.dep_ == 'rc' or node.head.dep_ == 'oc':
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
        # self.dictionary = pickle.load(open("plenar_dict.pkl", "rb"))
        self.dictionary = None
        self.res = []

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
        window_size = 15
        idf_weight = 1.0
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
            span = doc[start - window_size : end + window_size]
            spans.add(span)
        for span in spans:
            if span._.has_elite_neg and span._.has_volk:
                for token in span:
                    if token._.is_volk:
                        res["viz"].append(ContentAnalysis.get_viz(token, doclen, "V", idf_weight, dictionary=self.dictionary))
                        if token._.is_attr and token.i not in token_ids:
                            res["volk_attr"].append(token._.lemma)
                            token_ids.add((token.i, "V"))
                        else:
                            if token.i not in token_ids:
                                res["volk"].append(token._.lemma)
                                token_ids.add((token.i, "V"))

                        # sentence_start = span[0].sent.start
                        # sentence_end = span[-1].sent.end
                        # ps = ContentAnalysis.get_viz_start(doc, doc[sentence_start], f'PS | {ps_counter}')
                        # if ps not in res['viz']:
                        #     if last_start != sentence_start:
                        #         ps_counter += 1
                        #         res['viz'].append(ps)
                        #         last_start = sentence_start

                    if token._.is_elite_neg:
                        res["viz"].append(ContentAnalysis.get_viz(token, doclen, "E", idf_weight, dictionary=self.dictionary))
                        if token._.is_attr and token.i not in token_ids:
                            res["elite_attr"].append(token._.lemma)
                            token_ids.add((token.i, "E"))
                        else:
                            if token.i not in token_ids:
                                res["elite"].append(token._.lemma)
                                token_ids.add((token.i, "E"))

        res["viz"] = sorted(
            [dict(t) for t in {tuple(d.items()) for d in res["viz"]}],
            key=lambda i: i["start"],
        )
        # res["c_elite"] = Counter(res["elite"])
        # self.res["token_ids"] = token_ids
        res['doclen'] = doclen
        self.res.append(res)
        return doc

    # getters
    def has_elite_neg_getter(self, tokens):
        return any([t._.get("is_elite_neg") for t in tokens])

    def has_volk_getter(self, tokens):
        return any([t._.get("is_volk") for t in tokens])


    @staticmethod
    def get_viz(token, doclen, label, idf_weight, dictionary=None):
        start = token.idx
        end = token.idx + len(token.text) + 1
        token = token._.lemma
        if dictionary:
            score = ContentAnalysis.compute_score_per_term(token, doclen, idf_weight, dictionary)
        else:
            score = 0.0
        label = f"{label} | {score:.2f}"
        return {"start": start, "end": end, "label": label, "lemma": token}


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
    def compute_idf(term, dictionary, idf_weight=None):
        df = dictionary.dfs[dictionary.token2id[term.lower()]]
        if idf_weight:
            return tfidfmodel.df2idf(df, dictionary.num_docs, log_base=2.0, add=1.0)**idf_weight
        else:
            score = tfidfmodel.df2idf(df, dictionary.num_docs, log_base=2.0, add=1.0)
            return score


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
            score = compute_idf(i['lemma'], dictionary, idf_weight)
            label = i['label']
            i['label'] = label.replace(label.split('| ')[1], f'{score:.2f}')
        return viz


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


def content_analysis(directory, party="all", sample=None):
    if os.path.isdir(directory):
        print("Directory already exists.")
        return

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
    # c = ContentAnalysis("de_core_news_lg", "dict", directory=directory)
    # print(f"Pipeline with components: \n{'|'.join(nlp.pipe_names)}")
    # pipeline_params = {"content_analysis": {"window_size": 200}}
    for i, batch in enumerate(chunks(doc_labels, 5000)):
        nlp = spacy.load("de_core_news_lg")
        ca = ContentAnalysis(nlp)
        entity_recognizer = EntityRecognizer(nlp)
        sentiment_recognizer = SentimentRecognizer(nlp)
        nlp.add_pipe(ca, last=True)
        nlp.add_pipe(custom_lemma, before="content_analysis")
        nlp.add_pipe(sentiment_recognizer, before="content_analysis")
        nlp.add_pipe(entity_recognizer, before="content_analysis")
        nlp.remove_pipe("ner")
        labels = []
        for label in tqdm(batch):
            labels.append(label)
            if text:
                doc = nlp(text)
            else:
                doc = nlp(gendocs(label))
            # for token in doc:
            #     print(token.text, token.pos_, token.dep_, spacy.explain(token.dep_))
                # if token._.is_elite_neg:
                    # print('elite_neg', token.text)
        df = pd.DataFrame(ca.res)
        df['doc'] = labels
        # display(df)
        df.to_csv(f"res_ca/{directory}/res_ca_{party}_{i}.csv")
    print(f"Content Analysis complete. \nResults saved in {directory}/...")

    return df, labels


# %%
if __name__ == "__main__":
    df, labels = content_analysis('1125', party='all', sample=200)
    # ContentAnalysis.viz(gendocs(labels[0]), df.viz[0])
    # ContentAnalysis.viz(text, df.viz[0])

# %%
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

# %%
# always load first!
df = load_results_content_analysis('res_ca/1125')
# %%
dictionary = pickle.load(open("plenar_dict.pkl", "rb"))
compute_score_from_df(df, dictionary, 1.0)
# recount_viz(df, dictionary, 1.0)
evaluate_by_category('election_list', 'score', df)

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
viz = df.groupby(["party"]).resample("Q").mean().reset_index()
viz.drop(
    viz[(viz["party"] == "Parteilos") | (viz["party"] == "Die blaue Partei")].index,
    inplace=True,
)
viz2 = df.resample("Q").mean().reset_index()
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
