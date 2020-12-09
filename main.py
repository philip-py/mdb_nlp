# %%
# problemfälle: plenar_000786

# FINISH THE EXPOSE!

# good example: 'plenar_029688'
# eine automatische Analyse kann immer so gut sein wie die manuelle analyse, um sie zu validieren.

# %%
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
from src.d00_utils.helper import chunks, flatten, fix_umlauts, emb_fix_umlauts
from src.d01_ana import Results, MyCorpus, LossLogger,EndOfTraining, SentimentRecognizer, EntityRecognizer, ContentAnalysis
from src.d01_ana import load_data, gendocs, custom_extensions, recount_viz, compute_score_from_df, hash, intersect, merge_embeddings, load_models
from spacy_sentiws import spaCySentiWS
from math import fabs, log
from tqdm import tqdm
from gensim.models import tfidfmodel
from pathlib import Path
import pickle
import json
import random
import copy
from transformers import pipeline

def content_analysis(directory, party="all", sample=None, window_size=25, debug=False):

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
    print(f"Beginning Content Analysis with parameters: \n party: {party} | samplesize: {sample} | windowsize: {window_size}")
    nlp = spacy.load("de_core_news_lg")
    ca = ContentAnalysis(nlp, window_size=window_size)
    entity_recognizer = EntityRecognizer(nlp)
    sentiment_recognizer = SentimentRecognizer(nlp)
    sentiws = spaCySentiWS(sentiws_path='sentiws/')
    # clf = TextClassification(nlp)
    # nlp.add_pipe(custom_lemma, last=True)
    nlp.add_pipe(custom_extensions, last=True)
    nlp.add_pipe(sentiment_recognizer, last=True)
    nlp.add_pipe(sentiws, last=True)
    nlp.add_pipe(entity_recognizer, last=True)
    nlp.add_pipe(ca, last=True)
    # nlp.add_pipe(clf, last=True)
    nlp.remove_pipe("ner")
    labels = []
    for label in tqdm(doc_labels):
        labels.append(label)
        if text:
            doc = nlp(text)
            if debug:
                for token in doc:
                    print(token.text, token.ent_type_, token._.is_elite_neg, token._.is_attr, token._.is_negated, 'lemma', token._.lemma)
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

def discourse_analysis(directory, party=None, iter=1, sample=None, **kwargs):
    # %env PYTHONHASHSEED=0
    sns.set_style('darkgrid')

    logging.basicConfig(filename='w2v.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


    if not os.path.isdir(directory):
        print('Directory already exists.')
        return

    doc_labels = load_data(party)
    if sample:
        doc_labels = random.sample(doc_labels, sample)

    print('Number of documents: {}'.format(len(doc_labels)))
    sentences = MyCorpus(doc_labels)
    print(f'Beginning Discourse Analysis with parameters: \n{kwargs}')

    # model params
    model = Word2Vec(
    alpha=0.0025, min_alpha=0.00001, vector_size=300, epochs=300, seed=42, workers=8, hashfxn=hash, sorted_vocab=1, sg=1, hs=1, negative=0, sample=1e-4, **kwargs)

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
            loss_logger = LossLogger(party, i, directory)

            # train
            model.train(sentences, total_examples=total_examples, epochs=model.epochs, compute_loss=True, callbacks=[loss_logger])

        except EndOfTraining:
            print(f'End of Iteration: {i}')

    print(f'Discourse Analysis complete. \nResults saved in {directory}/...')


# %%
# content analysis
if __name__ == "__main__":
    dir = '1208_ws15'
    res = content_analysis(f'{dir}', party='all', window_size=15, sample=None)
    res.prepare(post=False)
    with open(f'res_ca/{dir}/results_all.pkl', 'wb') as f:
        pickle.dump(res, f)


# %%
# notebook
pd.set_option('display.max_rows', 25)

dir = '1208'
with open(f'res_ca/{dir}/results_all_post_ger2.pkl', 'rb') as f:
    res = pickle.load(f)

res.coding_pop(idf_weight=2.0)
res.df.sort_values('score', ascending=False).head(15)


# %%
res.top_spans(topn=5)

# %%
res.top_terms(party='DIE LINKE', abs=False, topn=20)

# %%
res.top_terms(party='AfD', abs=False, topn=20)

# %%
# timeseries visualization
def timeseries(res):
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

timeseries(res)


#%%
# regression
import statsmodels.api as sm

df_reg = res.df.copy()
df_reg = df_reg.loc["2013-10-01":"2020-01-01"]
reg = df_reg[["date", "score", "party", "birth_year"]].dropna()

reg["year"] = reg["date"].dt.strftime("%y")
reg["year"] = reg["year"].astype("int")
reg["year"] = reg["year"] - 13
reg["month"] = reg["date"].dt.strftime("%m")
reg["month"] = reg["month"].astype("int")
reg["is_summer"] = reg.apply(lambda row: sommerpause(row), axis=1)

sm.add_constant(reg)

regression = sm.Poisson.from_formula("score ~ C(party, Treatment('CDU')) + year", reg, missing="drop").fit()
# regression = sm.Poisson.from_formula("score ~ C(party, Treatment('CDU')) + year * C(party, Treatment('CDU'))", reg, missing="drop").fit()

sum = regression.summary()
sum









# %%
for hit in res.viz[res.labels.index('plenar_024197')]:
    # if hit['lemma'] == 'steuerzahler':
    if hit['span_start'] == 2376:
        print(hit['text'])




# %%
clf = pipeline("zero-shot-classification", model='joeddav/xlm-roberta-large-xnli', device=-1)

#%%
with open(f'res_ca/test/results_all.pkl', 'rb') as f:
    res = pickle.load(f)

# res_prepare(res, post=False)
res.viz = clf_pop(clf, res)

# %%
with open(f'res_ca/1203/results_all_post.pkl', 'wb') as f:
    pickle.dump(res, f)

# %%
res = pickle.load(open("res_ca/1203/results_all_clf.pkl", "rb"))
res.set_entities()
res.viz = coding(res)
res.compute_score(by_doclen=True, idf_weight=1.5, doclen_log=10, post=True)
res.compute_score_spans()
res.create_df()
res.add_meta_plenar()

with open(f'res_ca/1203/results_all_post.pkl', 'wb') as f:
    pickle.dump(res, f)

# res = pickle.load(open("res_ca/1203/results_all_post.pkl", "rb"))

# %%
res.viz = clf_pop_eu(clf, res)

# %%
with open(f'res_ca/test/results_all.pkl', 'wb') as f:
    pickle.dump(res, f)

# %%
text = 'Nun könnte man argumentieren, dass wir unser Engagement lieber einstellen und die Kosovaren sich selbst überlassen sollten . Dieser Gedanke mag verführerisch sein, weil er so einfach klingt . Das wäre aber verheerend für die Sicherheit und die Stabilität Europas . Meine Überzeugung ist . Der Westbalkan ist ein wunder Punkt mitten in der Europäischen Union . deshalb muss die EU gerade im Zeitalter globaler Krisen für eine europäische Zukunft des Westbalkans stärker politisch handeln . Versetzen wir uns einmal in die Lage des Kosovo . Das Kosovo und seine Bevölkerung befinden sich in einem permanenten psychologischen Zustand der Ungleichheit in seiner Region, zum Beispiel bei der Frage der Visaliberalisierung .'



# %%
res = pickle.load(open("res_ca/plenar/results_all_post_fix.pkl", "rb"))
res.prepare(post=False)

# %%
res.viz = coding(res)
res.compute_score(by_doclen=True, idf_weight=1.0, doclen_log=10, post=True)
res.compute_score_spans()
res.create_df()
res.add_meta_plenar()

pd.set_option('display.max_rows', 50)
res.df.sort_values('score', ascending=False).head(25)

# %%
span_sizes = []
for span in res.spans.values():
    for s in span.keys():
        span_sizes.append(s[1] - s[0])

max(span_sizes)




# %%
res_viz = clf_pop(res)
res.viz = res_viz
with open(f'res_ca/1201/results_all.pkl', 'wb') as f:
    pickle.dump(res, f)

# %%
res = pickle.load(open("res_ca/plenar/results_all_post_fix.pkl", "rb"))
res.set_entities()
res.viz = coding(res)
res.compute_score(by_doclen=True, idf_weight=1.0, doclen_log=10, post=True)
res.compute_score_spans()
res.create_df()
res.add_meta_plenar()
# res.top_spans(topn=5)
# res.visualize('plenar_002901', filter_by=['score', 'SPAN_IS_POP', 'TOK_IS_POP'])

pd.set_option('display.max_rows', 50)

res.df.sort_values('score', ascending=False).head(25)


# %%
n=6
res.visualize(res.top_spans()[n][0], res.top_spans()[n][1])

# %%
df = pd.read_json("data/plenar_meta.json", orient="index")
df.loc['plenar_025593.txt']
# %%
res.df.sort_values('score', ascending=False)

# %%
# res.visualize('plenar_024364', filter_by=False)
# res.visualize('plenar_002901', filter_by=False)
# res.visualize('plenar_002901', filter_by=['score', 'SPAN_IS_POP', 'RLY_E'])
# res.visualize('plenar_027415', filter_by=False)

# %%
res.top_spans(topn=5)

# %%
res.top_terms(party='DIE LINKE', abs=False, topn=20)

# %%
res.top_terms(party='AfD', abs=False, topn=20)

# %%
res.visualize('plenar_024364', span=(2320, 2908))

# %%
by_state = res.evaluate_by_category('election_list', 'score')
by_state

# %%
def timeseries(res):
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

timeseries(res)

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
# regression = sm.Poisson.from_formula("score ~ C(party, Treatment('CDU')) + year * C(party, Treatment('CDU'))", reg, missing="drop").fit()
# res = sm.OLS.from_formula("score ~ C(opp, Treatment('not_opp'))", reg, missing='drop').fit()
# res = sm.Poisson.from_formula("score ~ C(party, Treatment('CDU')) + date * C(party, Treatment('CDU'))", reg, missing='drop').fit()

sum = regression.summary()
sum

# %%
res.top_terms(party='AfD', abs=False, topn=20)
# res.top_terms(party='DIE LINKE', abs=False, topn=20)


# %%
# load embeddings
afd = merge_embeddings(load_models('AfD', iter=1))
linke = merge_embeddings(load_models('DIE LINKE', iter=1))
linke = emb_fix_umlauts(linke)

# %%
linke.wv.most_similar('bürger')

# %%
# save w2v-model to use gnsm-script
# python -m gensim.scripts.word2vec2tensor -i ~/gensim-data/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz
afd.wv.save_word2vec_format('model_afd', binary=False)















# %%
res.visualize('plenar_029688')
# res.visualize('plenar_029688', span=(3788, 4288))

# %%
res.visualize('plenar_029688', span=(6605, 7066))

# %%
res.df.groupby('party').mean()

# %%
res.df.sort_values(by='score', ascending=False).head()

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
    self.spans = span_dict

Results.compute_score_spans = compute_score_spans
# %%
def visualize(self, label, span=None):
    """visualize documents with displacy"""
    row = self.df.loc[self.df['doc'] == label].copy()
    text = gendocs(label)
    viz = self.viz[self.labels.index(label)].copy()

    if span:
        viz_span = []
        for hit in viz:
            if hit['span_start'] == span[0]:
                print(hit)
                # hit['start'] -= span[0]
                hit['end'] -= span[0]
                hit['label'] = f"{hit['label']} | {hit['score']:.2f}"
                viz_span.append(hit)
            ex = [
                {
                    "text": text[span[0]: span[1]],
                    "ents": viz_span,
                    "title": f"{row['doc'][0]} | {row.name_res[0]} ({row['party'][0]}) | {row['date'][0].strftime('%d/%m/%Y')}",
                }
            ]
        all_ents = {i["label"] for i in viz_span}
        # print(ex)

    else:
        for hit in viz:
            hit['label'] = f"{hit['label']} | {hit['score']:.2f}"
            ex = [
                {
                    "text": text,
                    "ents": viz,
                    "title": f"{row['doc'][0]} | {row.name_res[0]} ({row['party'][0]}) | {row['date'][0].strftime('%d/%m/%Y')}",
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

Results.visualize = visualize


# %%
res.visualize('plenar_029688', span=(3788, 4288))
# %%
all_spans = []
for doc in res.spans.items():
    for span in doc[1]:
        all_spans.append((doc[0], span, res.spans[doc[0]][span]))
all_spans.sort(key= lambda tup: tup[2], reverse=True)
all_spans[:]
# %%
res.spans['plenar_023592'][(1382, 1892)]

# %%

# %%
# with open(f'res_ca/1201/results_all_post.pkl', 'wb') as f:
#     pickle.dump(res, f)

def filter_res(res, label):
    new_res = Results()
    id = res.labels.index(label)
    new_res.viz = [res.viz[id]]
    new_res.labels = [res.labels[id]]
    new_res.doclens = [res.doclens[id]]
    new_res.scores = [res.scores[id]]
    new_res.spans = {label: res.spans[label]}
    return new_res

new = filter_res(res, 'plenar_019293')

# %%
sample = {'plenar_000566',
 'plenar_000786',
 'plenar_001144',
 'plenar_001333',
 'plenar_001338',
 'plenar_001354',
 'plenar_001403',
 'plenar_001640',
 'plenar_001810',
 'plenar_002320',
 'plenar_002731',
 'plenar_002875',
 'plenar_002876',
 'plenar_002879',
 'plenar_002886',
 'plenar_002898',
 'plenar_002899',
 'plenar_003242',
 'plenar_003389',
 'plenar_003455',
 'plenar_004466',
 'plenar_004774',
 'plenar_005146',
 'plenar_005247',
 'plenar_005340',
 'plenar_005365',
 'plenar_005464',
 'plenar_005540',
 'plenar_005780',
 'plenar_005786',
 'plenar_006131',
 'plenar_006160',
 'plenar_006208',
 'plenar_006257',
 'plenar_006556',
 'plenar_008097',
 'plenar_008306',
 'plenar_009392',
 'plenar_009633',
 'plenar_009754',
 'plenar_009927',
 'plenar_010388',
 'plenar_010464',
 'plenar_010902',
 'plenar_010971',
 'plenar_010984',
 'plenar_011029',
 'plenar_011165',
 'plenar_011264',
 'plenar_011616',
 'plenar_011661',
 'plenar_012804',
 'plenar_013082',
 'plenar_013291',
 'plenar_013881',
 'plenar_014181',
 'plenar_014534',
 'plenar_014938',
 'plenar_015053',
 'plenar_015136',
 'plenar_015164',
 'plenar_015532',
 'plenar_016888',
 'plenar_018397',
 'plenar_018916',
 'plenar_018989',
 'plenar_019489',
 'plenar_019594',
 'plenar_019608',
 'plenar_020077',
 'plenar_020452',
 'plenar_020469',
 'plenar_020833',
 'plenar_021241',
 'plenar_021338',
 'plenar_021351',
 'plenar_021806',
 'plenar_022467',
 'plenar_023300',
 'plenar_023619',
 'plenar_023654',
 'plenar_023932',
 'plenar_023945',
 'plenar_024197',
 'plenar_024198',
 'plenar_024208',
 'plenar_024215',
 'plenar_024347',
 'plenar_024797',
 'plenar_024988',
 'plenar_025574',
 'plenar_025824',
 'plenar_026708',
 'plenar_026711',
 'plenar_026854',
 'plenar_026899',
 'plenar_026913',
 'plenar_026981',
 'plenar_027078',
 'plenar_027223',
 'plenar_027334',
 'plenar_027396',
 'plenar_027401',
 'plenar_027439',
 'plenar_027471',
 'plenar_027615',
 'plenar_027616',
 'plenar_027738',
 'plenar_027787',
 'plenar_028394',
 'plenar_028452',
 'plenar_028679',
 'plenar_029108',
 'plenar_029119',
 'plenar_029366',
 'plenar_029571',
 'plenar_029673'}

#  ['plenar_024197', 'plenar_027439', 'plenar_029673', 'plenar_015136', 'plenar_029688']
