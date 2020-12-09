# %%
# import spacy
import pandas as pd
import pickle
import json
import random
import copy
# from collections import Counter
from transformers import pipeline
from pprint import pprint
# from germalemma import GermaLemma
# from spacy.matcher import Matcher
# from spacy.matcher import PhraseMatcher
# from spacy.tokens import Doc, Span, Token, DocBin
# from spacy.util import filter_spans
# from spacy.pipeline import EntityRuler
# from spacy import displacy
from src.d00_utils.helper import chunks, flatten
from src.d01_ana import Results
from src.d01_ana import gendocs
# from spacy_sentiws import spaCySentiWS
# from math import fabs, log
# from tqdm import tqdm
# from gensim.models import tfidfmodel
from pathlib import Path

# %%
res = pickle.load(open("res_ca/1201/results_all.pkl", "rb"))
res.set_entities()
res.compute_score()
res.compute_score_spans()
# sample
# n = 200
# res.viz = res.viz[:n]
# res.labels = res.labels[:n]
# res.doclens = res.doclens[:n]
# res.viz = res.viz[:10]
# res.create_df()
# res.add_meta_plenar()
# res.visualize(res.top_spans()[0][0])

# %%
def filter_res(res, label):
    new_res = Results()
    id = res.labels.index(label)
    new_res.viz = [res.viz[id]]
    new_res.labels = [res.labels[id]]
    new_res.doclens = [res.doclens[id]]
    new_res.scores = [res.scores[id]]
    new_res.spans = {label: res.spans[label]}
    return new_res

res_filter = filter_res(res, 'plenar_002901')

# %%
clf = pipeline("zero-shot-classification", model='joeddav/xlm-roberta-large-xnli', device=-1)

# %%
def clf_pop(res):
    # clf = pipeline("zero-shot-classification", model='joeddav/xlm-roberta-large-xnli', device=-1)
    res_viz = []
    for i, (doc, doc_vizs) in enumerate(zip(res.spans, res.viz)):
        if i % 500 == 0:
            print(i, f'/{len(res.spans)}')
        doc_viz = []
        doc_vizs = Results.filter_viz(doc_vizs, on='start')
        for span in res.spans[doc]:
            viz = []
            text = gendocs(doc)[span[0]:span[1]]
            viz.extend([viz for viz in doc_vizs if viz['span_start'] == span[0]])
            for v in viz:
                v['RLY_GER'] = True
                v['RLY_V'] = False
                v['RLY_E'] = False
                v['RLY_REASON'] = set()

            # 1. check if text is ger
            hypothesis_template = 'Der Text handelt von {}'
            candidate_labels = ['Deutschland', 'Europa', 'Ausland']
            s = clf(text, candidate_labels, hypothesis_template, multi_class=False)
            if s['labels'][0] == 'Ausland' and s['scores'][0] >= 0.9:
                for v in viz:
                    v['RLY_GER'] = False

            # 2. check if volk is benachteiligt:
            hypothesis_template = '{} hat Nachteile'
            candidate_labels = []
            for v in viz:
                if v['coding'] == 'V':
                    candidate_labels.append(v['lemma'])
            if hypothesis_template and candidate_labels:
                s = clf(text, candidate_labels, hypothesis_template, multi_class=True)

            candidates_people = []
            for j, label in enumerate(s['labels']):
                if s['scores'][j] >= 0.75:
                    candidates_people.append(label)
                    for v in viz:
                        if v['lemma'] == label:
                            v['RLY_V'] = True


            # 3. check if elite benachteiligt volk:
            for volk in candidates_people:
                h0 = '{} benachteiligt ' + volk
                h1 = '{} entmachtet ' + volk
                h2 = '{} betrügt ' + volk
                # h3 = '{} belügt ' + volk
                candidate_labels = []
                for v in viz:
                    if v['coding'] == 'E':
                        candidate_labels.append(v['lemma'])

                hs = [h0, h1, h2]
                for h, hypothesis_template in enumerate(hs):
                    if candidate_labels:
                        # print(hypothesis_template)
                        s = clf(text, candidate_labels, hypothesis_template, multi_class=True)
                        for j, label in enumerate(s['labels']):
                            if s['scores'][j] >= 0.75:
                                for v in viz:
                                    if v['lemma'] == label:
                                        v['RLY_E'] = True
                                        v['RLY_REASON'].add(h)

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
            # print(viz)
            doc_viz.extend(viz)
        res_viz.append(doc_viz)
    return res_viz


def coding(res):
    res_viz = []
    for i, (doc, doc_vizs) in enumerate(zip(res.spans, res.viz)):
        if i % 500 == 0:
            print(i, f'/{len(res.spans)}')
        doc_viz = []
        doc_vizs = Results.filter_viz(doc_vizs, on='start')
        for span in res.spans[doc]:
            viz = []
            text = gendocs(doc)[span[0]:span[1]]
            viz.extend([viz for viz in doc_vizs if viz['span_start'] == span[0]])

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
    return res_viz


# %%
res_filter.viz = clf_pop(res_filter)

# %%
res_filter.set_entities()
res_filter.create_df()
res_filter.add_meta_plenar()
res_filter.compute_score()
res_filter.visualize(res_filter.labels[0], filter_by=False)

# %%
res_viz = clf_pop(res)
res.viz = res_viz
with open(f'res_ca/1201/results_all_post_v2.pkl', 'wb') as f:
    pickle.dump(res, f)

# %%
res.compute_score()
res.compute_score_spans()
res.add_meta_plenar()
res.create_df()
res.top_spans(topn=5)

# %%
res.visualize('plenar_024364', span=(2320, 2908))

# %%
res.visualize('plenar_029586', span=(1450, 2257))

# %%
res.visualize('plenar_028414', span=(1723, 2193))

# %%
label = res.top_spans(topn=5)[4][0]
sp = res.top_spans(topn=5)[4][1]
res.visualize(label, sp)

# %%
# from germansentiment import SentimentModel
# clf = SentimentModel()
# clf = pipeline("sentiment-analysis", model='oliverguhr/german-sentiment-bert', device=-1)
clf = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

# %%
# implement sentiment??? add sentiment to token AND span?
def clf_sentiment(res):
    res_viz = []
    for i, (doc, doc_vizs) in enumerate(zip(res.spans, res.viz)):
        # print(i, f'/{len(res.spans)}')
        viz = []
        for span in res.spans[doc]:
            text = gendocs(doc)[span[0]:span[1]]
            # texts
            print(text)
            viz.extend([viz for viz in doc_vizs if viz['span_start'] == span[0]])
            for v in viz:
                v['SPAN_SENT'] = 0
            sents = text.strip().split('.')
            for sent in [i for i in sents if i]:
                s = clf(sent.lower())
                print(sent)
                print(s)
            print('-----------')
            t = clf(text)
            print(t)
            print()
            # if any()
            # result = (s['label'], s['score'])
            # if result[0] == 'negative':
            #     for v in viz:
            #         v['SPAN_SENT'] = -1
            # elif result[0] == 'neutral':
            #     for v in viz:
            #         v['SPAN_SENT'] = 0
            # elif result[0] == 'positive':
            #     for v in viz:
            #         v['SPAN_SENT'] = 1
            # print(s['label'], s['score'])
        res_viz.append(viz)
    return res_viz

res_viz = clf_sentiment(res)

# %%


# texts = [
#     "Mit keinem guten Ergebniss","Das ist gar nicht mal so gut",
#     "Total awesome!","nicht so schlecht wie erwartet",

# %%
texts = ["Jedes dritte Kind im Südsudan ist unterernährt, und eine Viertelmillion Kinder ist vom Hungertod bedroht .  Zudem berichtet UNICEF von einer immer schlimmer werdenden Gewalt gegen Kinder . Es gehört zur grausamen Kriegstaktik beider Parteien, gezielt Kinder zu vergewaltigen, zu verstümmeln und zu töten .  Unter den 80 Zivilisten, die im Oktober bei Kämpfen im Südsudan getötet wurden, waren mindestens 57 Kinder ."]
result = model.predict_sentiment(text)
print(result)

# %%
# after sentiment analysis of span:
# res.spans = list(res.spans)[:10]
# res.viz = res.viz[:10]
# res.labels = res.labels[:10]

        # check if wir is volk:
        # if "wir" in text.lower():
        #     hypothesis_template = 'wir sind {}'
        #     candidate_labels = ['partei']
        #     for v in res.viz[i]:
        #         if v['coding'] == 'V':
        #             candidate_labels.append(v['lemma'])
        #     if hypothesis_template and candidate_labels:
        #         s = clf(text, candidate_labels, hypothesis_template, multi_class=False)

        #     if s['scores'][0] >= 0.8:
        #         for v in res.viz[i]:
        #             if v['lemma'] == s['labels'][0]:
        #                 v['WIR'] = 1
        #                 print(v['lemma'], s['scores'][0])
        #                 print(text)
        #             else:
                        # v['WIR'] = 0


# %%
# res.labels = res.labels[:10]
# res.doclens = res.doclens[:10]
res.compute_score()
res.create_df()
# res.visualize('plenar_023955')

# %%
res.compute_score(post=True)
res.compute_score_spans()
# res.visualize(res.labels[-2])
res.create_df()
res.add_meta_plenar()
pprint(res.top_spans(topn=5))

# %%
res.visualize(res.top_spans()[0][0])


# %%
# Bürger ist Volk?
# volk = 'bürger'
# temp = f'{volk}'
# hypothesis_template = temp + 'ist {}'
# print(hypothesis_template)
# candidate_labels = ['Volk']

# elite negativ?
# elite = 'Regierung'
# temp = f"{elite}"
# hypothesis_template = temp + 'ist {}'
# print(hypothesis_template)
# candidate_labels = ['negativ']

# is deutsch?
# hypothesis_template = 'Der Text handelt von {}'
# print(hypothesis_template)
# candidate_labels = ['Deutschland', 'Ausland']

hypothesis_template = 'Wir sind {}'
print(hypothesis_template)
candidate_labels = ['Bevölkerung', 'Rechtspopulisten']


# hypothesis_template = '{} ist negativ'
# print(hypothesis_template)
# candidate_labels = ['Regierung', 'Bürger', 'Deutschland', 'Räuber', 'Automobilindustrie', 'Volk']
texts = ['Wenn Sie dazu auffordern, nicht die Probleme zu benennen, die sich in unserem Land stellen, wenn Sie dazu auffordern, die Ängste und Bedenken unserer Bevölkerung zu ignorieren, dann machen Sie genau das Gegenteil dessen, was wir eigentlich wollen .  Sie leiten Wasser auf die Mühlen der Rechtspopulisten, und Sie tragen mit dazu bei, dass sich unsere Bevölkerung weiter von der Politik abgrenzt .  Herr Bartsch, Sie fordern dazu auf, dass die CSU wieder in ihr Herkunftsland zurückkehren soll .  Ich bin der festen Überzeugung .']
# texts = ["Ich hasse diese Regierung. Wir sind dagegen."]
# texts = ['Im Gegensatz zur korrupten Regierung ist Merkel nicht so ok.']
for text in texts:
    s = clf(text, candidate_labels, hypothesis_template=hypothesis_template, multi_class = True)
    print(s)
    # print(s['score'])
    # print(s['label'])

# %%
texts = ['Angela Merkel wird als die schlechteste Kanzlerin in die Geschichte eingehen.']
for text in texts:
    hypothesis_template = 'Der Text beschreibt {}.'
    candidate_labels = ['Vergangenheit', 'Gegenwart']
    s = clf(text, candidate_labels, hypothesis_template, multi_class=False)
    print(s['labels'], s['scores'])
    if s['labels'][0] == 'Vergangenheit' and s['scores'][0] >= 0.9:
        # for v in res.viz[i]:
            # if v['span_start'] == span[0]:
                # v['RLY_GER'] = 0
        print(text)
    # if s['labels'][0] == 'Nationalsozialismus' and s['scores'][0] >= 0.9:
        # for v in res.viz[i]:
            # if v['span_start'] == span[0]:
                # v['RLY_GER'] = 0
        print(text)
    # else:
        # for v in res.viz[i]:
            # if v['span_start'] == span[0]:
                # v['RLY_GER'] = 1
# %%
# from src.d01_ana import Results
from src.d01_ana import gendocs
import pickle
res = pickle.load(open("res_ca/test/results_all.pkl", "rb"))
res.set_entities()
res.compute_score()
res.create_df()
res.add_meta_plenar()
# display(res.df.groupby('party').mean())
res.compute_score_spans()
# res.visualize('plenar_029688', span=(3788, 4288))

#%%
for i, (doc, _)  in enumerate(zip(res.spans, res.viz)):
    # print(doc, _)
    for span in res.spans[doc]:
        # decisions of transformers
        s = clf(gendocs(doc)[span[0]:span[1]])[0]
        print(gendocs(doc)[span[0]:span[1]])
        print(s['label'], s['score'])
        for v in res.viz[i]:
            if v['span_start'] == span[0]:
                v['span_sent'] = s['label']
                v['span_sent_score'] = s['score']

# %%
class ContentAnalysis():

    def __init__(self, model):
        self.nlp = spacy.load(model)
        self.clf = pipeline("zero-shot-classification", model='joeddav/xlm-roberta-large-xnli', device=-1)


        self.hypothesis_template = '{} sind korrupt'
        self.candidate_labels = ['poliker', 'manager', 'bänker']

        # custom lemmatizer
        # self.lemmatizer = GermaLemma()


    def analyze(self, text):

        result = []

        # res_dict = {'doc': label, 'len': None, 'pop': False, 'volk': 0, 'elite': 0, 'sents': None}
        # doc = nlp(gendocs(label))
        doc = self.nlp(text)
        # hits = {'volk': [], 'volk_text': [], 'elite': [], 'elite_text': [], 'attr': []}
        for i, sent in enumerate(doc.sents):
            # for j, token in enumerate(sent):
            print(sent.text)

            res = self.clf(sent.text, self.candidate_labels, hypothesis_template=self.hypothesis_template, multi_class=False)

            result.append(res)

        return result



# c = ContentAnalysis('de_core_news_lg')
c = ContentAnalysis('de_core_news_sm')

d = gendocs('data/corpus_plenar.txt')

# %%
res = []
texts = ['merkel ist korrupt', 'essen bleibt sehr lecker. Ich stimme voll und ganz zu']
# for text in list(d)[:200]:
    # res.append(c.analyze(text))





# %%
# SENTIMENT ANALYSIS
# %%
from transformers import pipeline
from pprint import pprint
# clf = pipeline("zero-shot-classification", model='joeddav/xlm-roberta-large-xnli', device=-1)
clf = pipeline("sentiment-analysis", model='oliverguhr/german-sentiment-bert', device=-1)

# %%
res = clf('''
    Parlamentarier sind meiner Ansicht neutral! Aber solange das sehr gute amerikanische Volk Bildzeitung lesen darf, ist alles ok. Darüber hinaus sollten wir unbedingt mehr Flüchtlinge abschieben. Vielen Dank meine lieben Damen und Herren.''')[0]
print(res['label'], res['score'])

# %%
from src.d01_ana import Results
from src.d01_ana import gendocs
import pickle
res = pickle.load(open("res_ca/test/results_all.pkl", "rb"))
res.set_entities()
res.compute_score()
res.create_df()
res.add_meta_plenar()
# display(res.df.groupby('party').mean())
res.compute_score_spans()
# res.visualize('plenar_029688', span=(3788, 4288))

#%%
for i, (doc, _)  in enumerate(zip(res.spans, res.viz)):
    # print(doc, _)
    for span in res.spans[doc]:
        # decisions of transformers
        s = clf(gendocs(doc)[span[0]:span[1]])[0]
        print(gendocs(doc)[span[0]:span[1]])
        print(s['label'], s['score'])
        for v in res.viz[i]:
            if v['span_start'] == span[0]:
                v['span_sent'] = s['label']
                v['span_sent_score'] = s['score']
        # if span is totally_pop
        # for v in res.viz[i]:
        #     if v['start'] == span[0]:
        #         v['span_is_pop'] = 1
        #     else:
        #         v['span_is_pop'] = 0

# %%
for hit in res.viz:
    print(hit)

# %%
