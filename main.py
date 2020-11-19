# %%
import os
from src.d01_ana.analysis import *
from src.d00_utils.helper import chunks, flatten

def compute_populism_score(dfs, dictionary):

    number_documents = len(dfs)

    dfs['score_pop'] = dfs.apply(lambda row: compute_pop_score(row.hits_pop, row['len'], dictionary, number_documents), axis=1)

    dfs['all_counter'] = dfs.apply(lambda row: {**row.volk_counter, **row.elite_counter}, axis=1)
    dfs['score'] = dfs.apply(lambda row: compute_score_from_counts(row.all_counter, row['len'], dictionary, number_documents), axis=1)
    dfs['score_volk'] = dfs.apply(lambda row: compute_score_from_counts(row.volk_counter, row['len'], dictionary, number_documents), axis=1)
    dfs['score_elite'] = dfs.apply(lambda row: compute_score_from_counts(row.elite_counter, row['len'], dictionary, number_documents), axis=1)

    dfs['score_pop_sum'] = dfs.apply(lambda row: compute_score_sum(row.score_pop), axis=1)

    return dfs


def content_analysis(directory, party='all', sample=None):

    if os.path.isdir(directory):
        print('Directory already exists.')
        return
    # party = 'all'

    # lemmatizer = GermaLemma()
    # nlp = spacy.load("de_core_news_lg")

    doc_labels = load_data(party)
    if sample:
        doc_labels = random.sample(doc_labels, sample)

    print('Number of documents: {}'.format(len(doc_labels)))

    print(f'Beginning Content Analysis with parameters: \n Party: {party}')

    c = ContentAnalysis('de_core_news_lg', 'dict', directory=directory)

    for i, batch in enumerate(chunks(doc_labels, 5000)):
        # print(f'Content Analysis on batch:{i+1}')
        res = []
        for label in tqdm(batch):
            res.append(c.analyze(label, gendocs(label)))
        df = pd.DataFrame(res)
        df.to_csv(f'res_ca/{directory}/res_ca_{party}_{i}.csv')

    print(f'Content Analysis complete. \nResults saved in {directory}/...')


def discourse_analysis(directory, party=None, iter=1, sample=None, **kwargs):

    logging.basicConfig(filename='w2v.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if os.path.isdir(directory):
        print('Directory already exists.')
        return

    Path(f'res_da/{directory}/').mkdir(parents=False, exist_ok=False)

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
if __name__ == "__main__":
    pass

# %%
%env PYTHONHASHSEED=0

# %%
discourse_analysis('afd_min2', party='AfD', iter=2, sample=None, min_count=2)

# %%
# content_analysis('plenar_all', party='AfD', sample=100)

# %%
print_doc('plenar_004996')

# %%
df = load_results_content_analysis('res_ca/results_ca')

# %%
def compute_from_counts(counts, doclen, dictionary, number_docs, idf_weight):
    scores = []
    for term, n in counts.items():
        score = compute_idf(term, dictionary, number_docs)
        scores.append(score*n)
        # print(term, n, score)
    # res = sum(scores) / log(doclen+10, 10)
    res = sum(scores)
    return res


def compute(row, dictionary, number_documents, idf_weight):
    if row['pop'] == True:
        volk = compute_from_counts(row.volk_counter, row.len, dictionary, number_documents, idf_weight)
        elite = compute_from_counts(row.elite_counter, row.len, dictionary, number_documents, idf_weight)
        attr = compute_from_counts(row.attr_counter, row.len, dictionary, number_documents, idf_weight)
    else:
        volk = 0.0
        elite = 0.0
        attr = 0.0
    return (volk, elite, attr)

def scoring(df, dictionary, idf_weight):
    number_documents = len(df)
    df['scores'] = df.apply(lambda row: compute(row, dictionary, number_documents, idf_weight), axis=1)

    # seperate into columns:
    df['score'] = df.apply(lambda row: sum(row.scores), axis=1)
    df['score_volk'] = df.apply(lambda row: row.scores[0], axis=1)
    df['score_elite'] = df.apply(lambda row: row.scores[1], axis=1)
    df['score_attr'] = df.apply(lambda row: row.scores[2], axis=1)



    return df

# %%
dictionary = pickle.load(open('plenar_dict.pkl', 'rb'))
df = scoring(df, dictionary, 10000)
df.groupby('party').mean()

# %%
import statsmodels.api as sm
df_reg = df.copy()
# df_reg = df_reg.loc[df['pop'] == True]
reg = df_reg[['typ', 'opp', 'date', 'score', 'party']].dropna()

reg['date'] = reg['date'].dt.strftime('%y')
reg['date'] = reg.date.astype('int')
reg['date'] = reg['date'] - 13

sm.add_constant(reg)

res = sm.Poisson.from_formula("score ~ C(party, Treatment('CDU')) + date", reg, missing='drop').fit()
# res = sm.Poisson.from_formula("score ~ C(party, Treatment('CDU')) + date * C(party, Treatment('CDU'))", reg, missing='drop').fit()

res.summary()

# %%
import plotly.express as px
viz = df.groupby(['party']).resample('Q').mean().reset_index()
viz.drop(viz[(viz['party'] == 'Parteilos') | (viz['party'] == 'Die blaue Partei')].index , inplace=True)
viz2 = df.resample('Q').mean().reset_index()
# viz.drop(viz[viz['party'] == 'Parteilos'], inplace=True)
# viz = df.resample('M').mean().reset_index()

# %%
import plotly.express as px
import plotly.graph_objects as go

fig = px.line(x=viz.date, y=viz.score, color=viz.party,
              title='Populism-Score')
fig.update_layout(width=1000,
                  title_font_size=20)

colors = ['darkblue', 'black', 'blue', 'green', 'magenta', 'gold', 'red']
for j, party in enumerate([i.name for i in fig.data]):
    fig.data[j].line.color=colors[j]
fig.add_trace(go.Scatter(x=viz2.date, y=viz2.score, mode="lines", name='Durchschnit / Quartal', marker_symbol="pentagon", line_width=5, line_color='darkgrey', line_dash='dash'))
fig.layout.template = "plotly_white"
fig.layout.legend.title.text = 'Partei'
fig.layout.xaxis.title.text = 'Jahr'
fig.layout.yaxis.title.text = 'Score'
fig.update_traces(hovertemplate='Score: %{y} <br>Jahr: %{x}')
for i in range(2,7):
    fig.data[i].visible = 'legendonly'
fig.show()

# print([i.name for i in fig.data])
# %%
dictionary = pickle.load(open('plenar_dict.pkl', 'rb'))
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
        scores.append(score*n)
    res = sum(scores) / log(doclen + 10, 10)
    return res

def finalise(row):
    if row['pop'] == False:
        return 0.0
    else:
        return sum([row.score_volk, row.score_elite, row.score_attr])

dictionary = pickle.load(open('plenar_dict.pkl', 'rb'))
# df = compute_pop_score_v2(df.sample(100), dictionary)
# df = df.sample(100)
number_documents = len(df)

df['attr_counter'] = df.apply(lambda row: Counter([term for i in row.elite_attr for term in (i[1] if i[1] is not None else [])]), axis=1)

df['score_volk'] = df.apply(lambda row: compute_score_from_counts_v2(row.volk_counter, row['len'], dictionary, number_documents), axis=1)
df['score_elite'] = df.apply(lambda row: compute_score_from_counts_v2(row.elite_counter, row['len'], dictionary, number_documents), axis=1)
df['score_attr'] = df.apply(lambda row: compute_score_from_counts_v2(row.attr_counter, row['len'], dictionary, number_documents), axis=1)

df['score'] = df.apply(lambda row: finalise(row), axis=1)
# df = df.loc[df['pop'] == True]
df.groupby('party').mean()


# %%
df = df[df['len'] > 800]

# %%
from sklearn.preprocessing import StandardScaler
sns.set_style('darkgrid')

scaler = StandardScaler()
scaler.fit(df[['score']])
df['zscore'] = scaler.transform(df[['score']])
# df['zscore_log'] = df.apply(lambda row: log(row['zscore']+1), axis=1)
# sns.violinplot(x=df['party'], y=df['score_pop_sum'], cut=0)
sns.violinplot(x=df['party'], y=df['score'], cut=0)
# sns.boxplot(x=df['party'], y=df['zscore'])

# %%
import statsmodels.api as sm
reg = df[['typ', 'opp', 'date', 'score', 'zscore', 'party']].dropna()

reg['date'] = reg['date'].dt.strftime('%y')
reg['date'] = reg.date.astype('int')
reg['date'] = reg['date'] - 13

sm.add_constant(reg)

# res = sm.Poisson.from_formula("score_pop_sum ~ C(opp, Treatment('not_opp')) + date + C(party, Treatment('CDU')) + date * C(party, Treatment('CDU'))", reg, missing='drop').fit_regularized()
# res = sm.Poisson.from_formula("score_pop_sum ~ date", reg).fit()

res = sm.OLS.from_formula("zscore ~ C(party, Treatment('CDU'))", reg, missing='drop').fit()

# res = sm.Poisson.from_formula("score ~ C(party, Treatment('CDU'))", reg, missing='drop').fit()

# res = sm.Poisson.from_formula("score ~ C(party, Treatment('CDU')) + date * C(party, Treatment('CDU'))", reg, missing='drop').fit()

# res = sm.OLS.from_formula("score ~ C(party, Treatment('CDU')) + date * C(party, Treatment('CDU'))", reg, missing='drop').fit()

# res = sm.Poisson.from_formula("score_pop ~ C(typ, Treatment('plenar')) + C(party, Treatment('CDU'))", reg, missing='drop').fit()
res.summary()

# %%
reg.groupby('party').mean()

# %%
