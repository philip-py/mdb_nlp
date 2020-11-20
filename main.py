# %%
import os
from src.d01_ana.analysis import *
from src.d00_utils.helper import chunks, flatten
from src.d01_ana.analysis_render import ContentAnalysisRender
from spacy import displacy
import re
import spacy
from collections import Counter


def content_analysis(directory, party="all", sample=None):
    if os.path.isdir(directory):
        print("Directory already exists.")
        return
    doc_labels = load_data(party)
    if sample:
        doc_labels = random.sample(doc_labels, sample)
    print("Number of documents: {}".format(len(doc_labels)))
    print(f"Beginning Content Analysis with parameters: \n Party: {party}")
    c = ContentAnalysis("de_core_news_lg", "dict", directory=directory)
    #
    for i, batch in enumerate(chunks(doc_labels, 5000)):
        res = []
        for label in tqdm(batch):
            res.append(c.analyze(label, gendocs(label), window_size=60))
        df = pd.DataFrame(res)
        df.to_csv(f"res_ca/{directory}/res_ca_{party}_{i}.csv")
    print(f"Content Analysis complete. \nResults saved in {directory}/...")


def discourse_analysis(directory, party=None, iter=1, sample=None, **kwargs):

    logging.basicConfig(
        filename="w2v.log",
        format="%(asctime)s : %(levelname)s : %(message)s",
        level=logging.INFO,
    )

    if os.path.isdir(directory):
        print("Directory already exists.")
        return

    Path(f"res_da/{directory}/").mkdir(parents=False, exist_ok=False)

    doc_labels = load_data(party)
    if sample:
        doc_labels = random.sample(doc_labels, sample)

    print("Number of documents: {}".format(len(doc_labels)))
    sentences = MyCorpus(doc_labels)
    print(f"Beginning Discourse Analysis with parameters: \n{kwargs}")

    # model params
    model = Word2Vec(
        alpha=0.0025,
        min_alpha=0.00001,
        vector_size=300,
        epochs=300,
        seed=42,
        workers=8,
        hashfxn=hash,
        sorted_vocab=1,
        sg=1,
        hs=1,
        negative=0,
        sample=1e-4,
        **kwargs,
    )

    model.build_vocab(sentences)

    # intersect
    pre = KeyedVectors.load("embeddings/wiki.model")
    res = intersect(pre, model)
    del pre

    model.wv.add_vectors(range(model.wv.vectors.shape[0]), res, replace=True)

    # ASSERT?

    total_examples = model.corpus_count

    for i in range(iter):
        try:
            seeds = random.choices(range(1_000_000), k=iter)
            seed = seeds[i]
            print(f"Seed in iteration {i}: {seed}")
            model.seed = seed
            loss_logger = LossLogger(party, i, directory)

            # train
            model.train(
                sentences,
                total_examples=total_examples,
                epochs=model.epochs,
                compute_loss=True,
                callbacks=[loss_logger],
            )

        except EndOfTraining:
            print(f"End of Iteration: {i}")

    print(f"Discourse Analysis complete. \nResults saved in {directory}/...")


def viz(label):
    """visualize documents with displacy"""
    # load all the things
    c = ContentAnalysisRender("de_core_news_lg", "dict", None, render=True)
    text = gendocs(label)
    res = c.analyze(label, text, window_size=50)
    df = pd.DataFrame([res])
    dfs = merge_meta(df)

    # fill dictionary for manual rendering
    ex = [
        {
            "text": text,
            "ents": res["viz"],
            "title": f"ID: {res['doc']} | {dfs.name_res[0]} ({dfs['party'][0]}) | {dfs['date'][0].strftime('%d/%m/%Y')}",
        }
    ]

    # find unique labels for coloring options
    all_ents = {i["label"] for i in res["viz"]}
    options = {"ents": all_ents, "colors": dict()}
    for ent in all_ents:
        if ent.startswith("E"):
            options["colors"][ent] = "coral"
        if ent.startswith("V"):
            options["colors"][ent] = "lightgrey"
        if ent.startswith("P"):
            options["colors"][ent] = "yellow"

    displacy.render(ex, style="ent", manual=True, jupyter=True, options=options)


def main():
    pass


if __name__ == "__main__":
    main()
    # %env PYTHONHASHSEED=0

    df = load_results_content_analysis("res_ca/results_ca")
    dictionary = pickle.load(open("plenar_dict.pkl", "rb"))

    df = populism_score_main(df, dictionary, 2.0)
    display(df.groupby("party").mean().sort_values('score', ascending=False))

# %%
def evaluate_by_category(category, target, df):
    grouped = df.groupby(category).mean().sort_values(target, ascending=False)
    mdbs_meta = pd.read_csv('data/mdbs_metadata.csv')
    res = pd.merge(per_name, mdbs_meta, how='left', on='name_res')
    return res

by_name = evaluate_by('name_res', 'score', df)
by_name[:100].party.value_counts()

# %%
# EVALUATE
eval = df.sort_values("score", ascending=False)[["party", "doc"]][:500]
labels = [i for i in eval.doc]

for label in labels[:3]:
    viz(label)

# %%
# what to do?
# discourse_analysis('afd_min2', party='AfD', iter=2, sample=None, min_count=2)
# content_analysis('plenar_all', party='AfD', sample=100)
# single_label = load_data('AfD')[666]
# viz(single_label)


# %%

# %%
def top_terms(df, cat, party=None):
    if party:
        df = df.loc[df.party == party]
    return sum([i for i in df[cat]], Counter())


def top_terms_score(counts, dictionary):
    """
    doesnt take into account len of document right now.
    """
    res = []
    for k, v in counts.items():
        score = compute_idf(k, dictionary, idf_weight=2.0) * v
        res.append((k, score))
    res.sort(key=lambda x: x[1], reverse=True)
    return res


dictionary = pickle.load(open("plenar_dict.pkl", "rb"))
top_absolute = top_terms(df, "attr_counter", party="SPD")
top_idf = top_terms_score(top_absolute, dictionary)

top_idf

# %%
df = populism_score_main(df, dictionary, 2.0)
df.groupby("party").mean()

# %%


# %%
import statsmodels.api as sm

df_reg = df.copy()
# change daterange?
df_reg = df_reg.loc["2018-01-01":"2020-01-01"]
# df_reg = df_reg.loc[df['pop'] == True]
reg = df_reg[["typ", "opp", "date", "score", "party"]].dropna()


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

res = sm.Poisson.from_formula(
    "score ~ C(party, Treatment('CDU')) + year", reg, missing="drop"
).fit()
# res = sm.OLS.from_formula("score ~ C(opp, Treatment('not_opp'))", reg, missing='drop').fit()
# res = sm.Poisson.from_formula("score ~ C(party, Treatment('CDU')) + date * C(party, Treatment('CDU'))", reg, missing='drop').fit()

sum = res.summary()

# %%
df.sort_values("score", ascending=False)[["party", "doc"]][:500]

# %%


# %%
import plotly.express as px

viz = df.groupby(["party"]).resample("Q").mean().reset_index()
viz.drop(
    viz[(viz["party"] == "Parteilos") | (viz["party"] == "Die blaue Partei")].index,
    inplace=True,
)
viz2 = df.resample("Q").mean().reset_index()
# viz.drop(viz[viz['party'] == 'Parteilos'], inplace=True)
# viz = df.resample('M').mean().reset_index()

# %%
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
dictionary = pickle.load(open("plenar_dict.pkl", "rb"))
df = compute_populism_score(df, dictionary)
# %%
"""
How to improve?
- if pop = True -> count all anti-elitist statements
- count only the first/the highest volks-anrufung for each pop_sent (bürgerinnen & bürger nicht als zwei zählen!)
- when to log?
- use absolute numbers per speech? Also nicht durch length teilen!
- sum at the end (each pop_sent + each elite_neg)
- make sure score is computed correctly
- create a very small evaluation set!
"""

# %%
models = load_models("ALL", 3)
model = merge_embeddings(models)

# %%
model.wv.most_similar("migration")

# %%
def compute_res(df, dictionary):

    number_documents = len(df)

    # dfs['score_pop'] = dfs.apply(lambda row: compute_pop_score(row.hits_pop, row['len'], dictionary, number_documents), axis=1)

    # dfs['all_counter'] = dfs.apply(lambda row: {**row.volk_counter, **row.elite_counter}, axis=1)
    # dfs['score'] = dfs.apply(lambda row: compute_score_from_counts(row.all_counter, row['len'], dictionary, number_documents), axis=1)
    # dfs['score_volk'] = dfs.apply(lambda row: compute_score_from_counts(row.volk_counter, row['len'], dictionary, number_documents), axis=1)
    # dfs['score_elite'] = dfs.apply(lambda row: compute_score_from_counts(row.elite_counter, row['len'], dictionary, number_documents), axis=1)

    # dfs['score_pop_sum'] = dfs.apply(lambda row: compute_score_sum(row.score_pop), axis=1)

    return df


def compute_pop_score_v2(all_hits, doclen, dictionary, number_docs):
    res = dict()
    scores_volk = []
    scores_elite = []
    for hit in all_hits:
        for volk in hit[0]:
            term = volk[0]
            idf = compute_idf(term, dictionary, number_docs)
            score = idf / log(doclen + 10, 10)
            scores_volk.append(score)

        for elite in hit[1]:
            term = elite[0]
            idf = compute_idf(term, dictionary, number_docs)
            score = idf / log(doclen + 10, 10)
            scores_elite.append(score)

    res["volk"] = sum(scores_volk)
    res["elite"] = sum(scores_elite)

    return res


def compute_score_from_counts_v2(counts, doclen, dictionary, number_docs):
    scores = []
    for term, n in counts.items():
        # df = dictionary.dfs[dictionary.token2id[term.lower()]]
        # score = tfidfmodel.df2idf(df, number_docs, log_base=2.0, add=1.0)
        score = compute_idf(term, dictionary, number_docs)
        scores.append(score * n)
    res = sum(scores) / log(doclen + 10, 10)
    return res


def finalise(row):
    if row["pop"] == False:
        return 0.0
    else:
        return sum([row.score_volk, row.score_elite, row.score_attr])


dictionary = pickle.load(open("plenar_dict.pkl", "rb"))
# df = compute_pop_score_v2(df.sample(100), dictionary)
# df = df.sample(100)
number_documents = len(df)

df["attr_counter"] = df.apply(
    lambda row: Counter(
        [term for i in row.elite_attr for term in (i[1] if i[1] is not None else [])]
    ),
    axis=1,
)

df["score_volk"] = df.apply(
    lambda row: compute_score_from_counts_v2(
        row.volk_counter, row["len"], dictionary, number_documents
    ),
    axis=1,
)
df["score_elite"] = df.apply(
    lambda row: compute_score_from_counts_v2(
        row.elite_counter, row["len"], dictionary, number_documents
    ),
    axis=1,
)
df["score_attr"] = df.apply(
    lambda row: compute_score_from_counts_v2(
        row.attr_counter, row["len"], dictionary, number_documents
    ),
    axis=1,
)

df["score"] = df.apply(lambda row: finalise(row), axis=1)
# df = df.loc[df['pop'] == True]
df.groupby("party").mean()


# %%
df = df[df["len"] > 800]

# %%
dictionary = pickle.load(open("plenar_dict.pkl", "rb"))
# %%
from sklearn.preprocessing import StandardScaler

sns.set_style("darkgrid")

scaler = StandardScaler()
scaler.fit(df[["score"]])
df["zscore"] = scaler.transform(df[["score"]])
# df['zscore_log'] = df.apply(lambda row: log(row['zscore']+1), axis=1)
# sns.violinplot(x=df['party'], y=df['score_pop_sum'], cut=0)
sns.violinplot(x=df["party"], y=df["score"], cut=0)
# sns.boxplot(x=df['party'], y=df['zscore'])

# %%
import statsmodels.api as sm

reg = df[["typ", "opp", "date", "score", "zscore", "party"]].dropna()

reg["date"] = reg["date"].dt.strftime("%y")
reg["date"] = reg.date.astype("int")
reg["date"] = reg["date"] - 13

sm.add_constant(reg)

# res = sm.Poisson.from_formula("score_pop_sum ~ C(opp, Treatment('not_opp')) + date + C(party, Treatment('CDU')) + date * C(party, Treatment('CDU'))", reg, missing='drop').fit_regularized()
# res = sm.Poisson.from_formula("score_pop_sum ~ date", reg).fit()

res = sm.OLS.from_formula(
    "zscore ~ C(party, Treatment('CDU'))", reg, missing="drop"
).fit()

# res = sm.Poisson.from_formula("score ~ C(party, Treatment('CDU'))", reg, missing='drop').fit()

# res = sm.Poisson.from_formula("score ~ C(party, Treatment('CDU')) + date * C(party, Treatment('CDU'))", reg, missing='drop').fit()

# res = sm.OLS.from_formula("score ~ C(party, Treatment('CDU')) + date * C(party, Treatment('CDU'))", reg, missing='drop').fit()

# res = sm.Poisson.from_formula("score_pop ~ C(typ, Treatment('plenar')) + C(party, Treatment('CDU'))", reg, missing='drop').fit()
res.summary()

# %%
reg.groupby("party").mean()
