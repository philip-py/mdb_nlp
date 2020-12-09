# %%
from transformers import pipeline
import pickle
from src.d01_ana import Results, gendocs
from pprint import pprint
import pandas as pd


def res_prepare(res, post=False):
    res.set_entities()
    res.compute_score(by_doclen=True, idf_weight=1.5, doclen_log=10, post=post)
    res.compute_score_spans()
    res.create_df()
    res.add_meta_plenar()


def clf_pop(clf, res, debug=False):
    # clf = pipeline("zero-shot-classification", model='joeddav/xlm-roberta-large-xnli', device=-1)
    res_viz = []
    for i, (doc, doc_vizs) in enumerate(zip(res.spans, res.viz)):
        if i % 500 == 0:
            print(i, f'/{len(res.spans)}')
        doc_viz = []
        # doc_vizs = Results.filter_viz(doc_vizs, on='start')
        for span in res.spans[doc]:
            viz = []
            text = gendocs(doc)[span[0]:span[1]]
            # viz.extend([viz for viz in doc_vizs if viz['span_start'] == span[0] and viz['start'] - viz['span_start'] <= 2_400] and viz['RLY_GER'])
            viz.extend([viz for viz in doc_vizs if viz['span_start'] == span[0] and viz['span_end'] == span[1] and viz['RLY_GER']])
            for v in viz:
                v['RLY_V'] = False
                v['RLY_E'] = False
                v['RLY_REASON'] = set()

            # 2. check if volk is benachteiligt:
            condition = False
            while not condition:
                h0 = '{} hat Nachteile'
                # h1 = 'ungerecht für {}'
                candidate_labels = set()
                for v in viz:
                    if v['coding'] == 'V':
                        candidate_labels.add(v['lemma'])
                candidate_labels = list(candidate_labels)
                hs = [h0]
                for h, hypothesis_template in enumerate(hs):
                    if hypothesis_template and candidate_labels:
                        s = clf(text, candidate_labels, hypothesis_template, multi_class=True)

                    candidates_people = []
                    for j, label in enumerate(s['labels']):
                        if s['scores'][j] >= 0.75:
                            candidates_people.append(label)
                            for v in viz:
                                if v['lemma'] == label:
                                    v['RLY_V'] = True
                                    v['RLY_REASON'].add(h)
                            condition = True

                        if debug:
                            pprint(hypothesis_template)
                            pprint(s)

                condition = True


            # 3. check if elite benachteiligt volk:
            for volk in candidates_people:
                condition = False
                while not condition:
                    h0 = '{} benachteiligt ' + volk
                    h1 = '{} entmachtet ' + volk
                    h2 = '{} betrügt ' + volk
                    # h3 = '{} belügt ' + volk
                    candidate_labels = set()
                    for v in viz:
                        if v['coding'] == 'E' or (v['coding'] == 'EA' and v['pos'] == 'NOUN'):
                            candidate_labels.add(v['lemma'])
                    candidate_labels = list(candidate_labels)

                    hs = [h0, h1, h2]
                    for h, hypothesis_template in enumerate(hs):
                        if candidate_labels:
                            s = clf(text, candidate_labels, hypothesis_template, multi_class=True)
                            for j, label in enumerate(s['labels']):
                                if s['scores'][j] >= 0.75:
                                    for v in viz:
                                        if v['lemma'] == label:
                                            v['RLY_E'] = True
                                            v['RLY_REASON'].add(h)
                                    condition=True

                            if debug:
                                pprint(hypothesis_template)
                                pprint(s)
                    condition=True

            doc_viz.extend(viz)
        res_viz.append(doc_viz)
    return res_viz


def clf_ger(clf, res, debug=False):
    res_viz = []
    for i, (doc, doc_vizs) in enumerate(zip(res.spans, res.viz)):
        if i % 500 == 0:
            print(i, f'/{len(res.spans)}')
        doc_viz = []
        seen_span = set()
        # doc_vizs = Results.filter_viz(doc_vizs, on='start')
        for span in res.spans[doc]:
            viz = []
            span_id = (span[0], span[1])
            text = gendocs(doc)[span[0]:span[1]]
            # viz.extend([viz for viz in doc_vizs if viz['span_start'] == span[0] and viz['start'] - viz['span_start'] <= 2_400])

            if span_id not in seen_span:
                viz.extend([viz for viz in doc_vizs if viz['span_start'] == span[0] and viz['span_end'] == span[1]])
                seen_span.add(span_id)

            for v in viz:
                v['RLY_GER'] = True

            # if viz:
            # 1. check if text is ger
            hypothesis_template = 'Der Text handelt von {}'
            candidate_labels = ['Deutschland', 'Europa', 'Ausland']
            s = clf(text, candidate_labels, hypothesis_template, multi_class=True)
            # if s['labels'][0] == 'Ausland' and s['scores'][0] > 0.5:
            id_ausland = s['labels'].index('Ausland')
            id_ger = s['labels'].index('Deutschland')
            if s['labels'][-1] == 'Deutschland' and s['scores'][id_ausland] > 0.5:
                for v in viz:
                    v['RLY_GER'] = False

            elif s['labels'][0] == 'Ausland' and s['scores'][id_ausland] / s['scores'][id_ger] >  2:
                for v in viz:
                    v['RLY_GER'] = False

            ######################################
            # 1. check if text is ger v2:
            # hypothesis_template = 'Der Text beschreibt {}'
            # candidate_labels = ['Deutschland', 'Ausland']
            # s = clf(text, candidate_labels, hypothesis_template, multi_class=False)
            # if s['labels'][0] == 'Ausland' and s['scores'][0] >= 0.9:
            #     for v in viz:
            #         v['RLY_GER'] = False
            #####################################

            if debug:
                pprint(span_id)
                pprint(hypothesis_template)
                pprint(s)

            doc_viz.extend(viz)
        res_viz.append(doc_viz)
    return res_viz


def clf_demo(clf, res, debug=False):
    res_viz = []
    for i, (doc, doc_vizs) in enumerate(zip(res.spans, res.viz)):
        if i % 500 == 0:
            print(i, f'/{len(res.spans)}')
        doc_viz = []
        # doc_vizs = Results.filter_viz(doc_vizs, on='start')
        for span in res.spans[doc]:
            viz = []
            text = gendocs(doc)[span[0]:span[1]]
            # viz.extend([viz for viz in doc_vizs if viz['span_start'] == span[0] and viz['span_end'] == span[1]])
            # viz.extend([viz for viz in doc_vizs if viz['span_start'] == span[0] and viz['start'] - viz['span_start'] <= 2_400])
            viz.extend([viz for viz in doc_vizs if viz['span_start'] == span[0] and viz['span_end'] == span[1] and viz['RLY_GER']])
            checked_history = False
            is_present = True
            demo = ['Demokratie', 'Gewaltenteilung', 'Gerechtigkeit', 'Meinungsfreiheit']
            for w in demo:
                if w in text:
                    if not checked_history:
                        hypothesis_template = 'Der Text beschreibt {}'
                        candidate_labels = ['Geschichte', 'Nationalsozialismus']
                        s = clf(text, candidate_labels, hypothesis_template, multi_class=True)
                        if debug:
                            print(s)
                        if any(i > 0.75 for i in s['scores']):
                            is_present=False
                            checked_history=True

                    if is_present:
                        # REASON IS S
                        hypothesis_template = 'In Deutschland herrscht keine {}'
                        candidate_labels = [w]
                        s = clf(text, candidate_labels, hypothesis_template, multi_class=True)
                        if s['scores'][0] > 0.75:
                            for v in viz:
                                if v['coding'].startswith('E'):
                                    v['RLY_E'] = True
                                    v['RLY_REASON'].add('S')
                                elif v['coding'].startswith('V'):
                                    v['RLY_V'] = True
                                    v['RLY_REASON'].add('S')

                        if debug:
                            pprint(hypothesis_template)
                            pprint(s)

            doc_viz.extend(viz)
        res_viz.append(doc_viz)
    return res_viz


# %%
if __name__ == "__main__":
    clf = pipeline("zero-shot-classification", model='joeddav/xlm-roberta-large-xnli', device=-1)

# %%
######  FOLDER  ##############
folder = '1209' ##############
##############################

# %%
with open(f'res_ca/{folder}/results_all.pkl', 'rb') as f:
    res = pickle.load(f)

print('clf GER')
res.viz = clf_ger(clf, res, debug=False)
print('clf POP')
res.viz = clf_pop(clf, res, debug=False)
print('clf DEMO')
res.viz = clf_demo(clf, res, debug=False)

with open(f'res_ca/{folder}/results_all_post.pkl', 'wb') as f:
    pickle.dump(res, f)

#%%
dir = '1209'
with open(f'res_ca/{dir}/results_all_post.pkl', 'rb') as f:
    res = pickle.load(f)

pd.set_option('display.max_rows', 25)
res.coding_pop(idf_weight=2.0)
res.df.sort_values('score', ascending=False).head(25)
























# %%
with open(f'res_ca/{folder}/results_all_post_demo.pkl', 'rb') as f:
    res = pickle.load(f)
res = coding_post(res)

#%%
res = coding_post(res)
pd.set_option('display.max_rows', 25)
res.df.sort_values('score', ascending=False).head(25)


# %%
test = res.filter_res('plenar_005340')

test.viz = clf_ger(clf, test, debug=True)






# %%
with open(f'res_ca/{folder}/results_all_post_demo.pkl', 'rb') as f:
    res = pickle.load(f)

print('clf GER')
res.viz = clf_ger(clf, res, debug=False)

with open(f'res_ca/{folder}/results_all_post_ger2.pkl', 'wb') as f:
    pickle.dump(res, f)



# %%
folder = 1208
with open(f'res_ca/{folder}/results_all_post.pkl', 'rb') as f:
    res = pickle.load(f)
res = do_post(res)

folder = 'test_1208'
with open(f'res_ca/{folder}/results_all_post.pkl', 'rb') as f:
    res2 = pickle.load(f)
res2 = do_post(res2)

# %%
# compare
display(res.df.loc[res.df.doc == 'plenar_001354'])
display(res2.df.loc[res2.df.doc == 'plenar_001354'])
display(res.df.loc[res.df.doc == 'plenar_027396'])
display(res2.df.loc[res2.df.doc == 'plenar_027396'])


# %%
res = do_post(res)
res2 = do_post(res2)


# %%
res2 = res
res2.viz = clf_demo(clf, res, debug=False)
# with open(f'res_ca/{folder}/results_all_post.pkl', 'wb') as f:
    # pickle.dump(res, f)

# %%
res2.set_entities()
res2.viz = coding(res2)
res2.compute_score(by_doclen=True, idf_weight=2.0, doclen_log=10, post=True)
res2.create_df()
res2.add_meta_plenar()

	# plenar_001354
# %%
pd.set_option('display.max_rows', 25)
res2.df.sort_values('score', ascending=False).head(25)

# %%
[i for i in res.df.sort_values('score', ascending=False).head(50).doc]
# %%
res2.visualize('plenar_002731')

#%%
with open(f'res_ca/{folder}/results_all_post.pkl', 'rb') as f:
    res = pickle.load(f)

def coded(res, label, index_start, categories=None):
    for hit in res.viz[res.labels.index(label)]:
        # if hit['lemma'] == 'steuerzahler':
        if hit['span_start'] == index_start:
            if not categories:
                pprint(hit)
            else:
                pprint({cat: hit[cat] for cat in categories})

coded(res, 'plenar_024197', 2376, categories=['lemma', 'RLY_GER', 'RLY_V', 'RLY_E', 'RLY_REASON', 'coding'])

# %%
coded(res, 'plenar_020833', 12886, categories=['lemma', 'RLY_GER', 'RLY_V', 'RLY_E', 'RLY_REASON', 'coding'])



# %%
new = filter_res(res, 'plenar_024197')
new.prepare(post=True)

# %%
new.viz = clf_pop(clf, new, debug=True)

# %%
new.viz = coding(new)
new.compute_score(by_doclen=True, idf_weight=1.5, doclen_log=10, post=True)
new.set_entities()
new.prepare(post=True)
new.compute_score_spans()
new.create_df()
new.add_meta_plenar()
new.visualize('plenar_024197')











# %%
# double check scores
score = 0
seen = set()
for hit in res.viz[res.labels.index('plenar_024197')]:
    if hit['start'] not in seen:
        seen.add(hit['start'])
    # if hit['lemma'] == 'steuerzahler':
    if hit['span_start'] == 2547:
        pprint(hit)
    score += hit['score']

score








# %%
folder = '1206'
with open(f'res_ca/{folder}/results_all_post_v2.pkl', 'rb') as f:
    res = pickle.load(f)

# %%
new = filter_res(res, 'plenar_024197')
new.prepare(post=True)

# %%
new.viz = clf_pop(clf, new, debug=True)

# %%
new.visualize('plenar_024197')

# %%
with open(f'res_ca/{folder}/results_all_post_v2.pkl', 'wb') as f:
    pickle.dump(res, f)



# %%
new.prepare(post=False)

# %%
new.viz = clf_pop_eu(clf, new)

# %%
text = 'dauerhafte Vergemeinschaftung der Schulden, ein Euro Finanzminister mit einem eigenen Budget hauptsächlich finanziert vom deutschen Steuerzahler, versteht sich und ein eigenes Euro Zonen Parlament . Von Gewaltenteilung ist überhaupt gar keine Spur mehr . Sehr geehrte Damen und Herren, das ist ein Skandal, und Sie haben das zu verantworten . Der Euro sollte dazu führen, dass Europa zusammenwächst . Von einem wahren Friedensprojekt sprach einst Helmut Kohl .'
hypothesis_template = 'In Deutschland herrscht keine {}'
candidate_labels = ['Demokratie', 'Gewaltenteilung']
print(hypothesis_template)
s = clf(text, candidate_labels, hypothesis_template, multi_class=True)
pprint(s)
# %%
demo = ['Demokratie', 'Gewaltenteilung', 'Gerechtigkeit']

# %%
# any([True for e in demo if e in text])
for w in demo:
    if w in text:
        print(w)
# %%
