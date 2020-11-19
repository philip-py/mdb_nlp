# %%
from germalemma import GermaLemma

def analysis(doc_labels):

    def gendocs(label):
        with open('data/corpus_clean/{}.txt'.format(label), "r") as text_file:
            return text_file.read()

    # %%

    # %%
    nlp = spacy.load("de_core_news_lg")

    # %%

    lemmatizer = GermaLemma()

    def lemma_getter(token):
        try:
            return lemmatizer.find_lemma(token.text, token.tag_)
        except:
            return token.lemma_

    def is_neg_elite(token):
        global found

        if token._.is_elite_noneg:
            found.append((token.text, None))
            return True

        elif token._.is_elite:
            check = list(token.children)
            # if token.head:
            #     check.append(token.head)
            node = token
            while node.head:
                seen = node
                if seen == node.head:
                    break
                else:
                    check.append(node)
                    node = seen.head
            attr_neg = [child for child in check if child._.lemma.lower() in negativ]
            if attr_neg:
                found.append((token.text, attr_neg))
                return True
            else:
                return False
            # return any([True for child in check if child._.lemma.lower() in negativ])
        else:
            return False


    def is_volk(token):
        global found
        # if token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
            # print(token._.lemma)

        check = list(token.children)

        if token._.lemma.lower() in people:
            found.append((token.text, None))
            return True

        elif token._.lemma.lower() in people_ordinary:
            attr_ppl = [child for child in check if child._.lemma.lower() in attribut_ordinary]
            if attr_ppl:
                found.append((token.text, attr_ppl))
                # print('found attr_ppl')
                return True
            else:
                return False

        elif token._.lemma.lower() in people_ger:
            attr_ger = [child for child in check if child._.lemma.lower() in attribut_ger]
            if attr_ger:
                found.append((token.text, attr_ger))
                # print('found ppl_ger')
                return True
            else:
                return False

        else:
            return False


    people = set(people)
    people_ordinary = set(people_ordinary)
    people_ger = set(people_ger)
    attr_ger = set(attribut_ger)
    elite = [*elite_pol, *elite_eco, *elite_experten, *elite_medien]
    elite = set(elite)
    elite_noneg = set(elite_noneg)

    negativ = set(neg_dict.keys())
    positiv = set(pos_dict.keys())

    dfs = []
    all_sents = []
    res = []

    # doc_labels = doc_labels[1000:1500]
    # doc_labels = random.sample(doc_labels, 100)

    for label in tqdm(doc_labels):

        res_dict = {'doc': None, 'len': None, 'pop': False, 'volk': 0, 'elite': 0, 'sents': None, 'volk_': None, 'elite_': None, 'lemma_pop': None}

        found = []
        doc = nlp(gendocs(label))
        hits = {'volk': [], 'elite': []}
        for i, sent in enumerate(doc.sents):
            # print(sent)
            for j, token in enumerate(sent):
                # is_volk_getter = lambda token: token._.lemma.lower() in volk
                is_elite_getter = lambda token: token._.lemma.lower() in elite
                is_elite_noneg_getter = lambda token: token._.lemma.lower() in elite_noneg
                is_neg_getter = lambda token: token._.lemma.lower() in negativ
                is_pos_getter = lambda token: token._.lemma.lower() in positiv

                Token.set_extension('is_neg', getter=is_neg_getter, force=True)
                Token.set_extension('is_pos', getter=is_pos_getter, force=True)
                Token.set_extension('is_elite', getter=is_elite_getter, force=True)
                Token.set_extension('is_elite_noneg', getter=is_elite_noneg_getter, force=True)
                Token.set_extension('lemma', getter=lemma_getter, force=True)

                is_volk_getter = lambda token: is_volk(token)
                is_neg_elite_getter = lambda token: is_neg_elite(token)

                Token.set_extension('is_volk', getter = is_volk_getter, force=True)
                Token.set_extension('is_neg_elite', getter = is_neg_elite_getter, force=True)

                if token._.is_volk:
                    hits['volk'].append(token._.lemma)

                if token._.is_neg_elite:
                    hits['elite'].append(token._.lemma)
                    all_sents.append(sent)

                # Token.set_extension('is_pos_volk', getter=is_pos_volk_getter_func, force=True)

                # print(token.text, token.lemma_, token._.lemma, token.pos_)
                # print(list(token.children))
        # print(found)

        matcher = Matcher(nlp.vocab)
        pattern = [{'_': {'is_neg_elite': True}}]
        matcher.add('text', None, pattern)
        matches = matcher(doc)
        has_pop = []
        tokens_pop = []
        for match_id, start, end in matches:
            span = doc[start-280:end+280]

            for token in span:
                if token._.is_volk:

                    tokens_pop.append(doc[start]._.lemma)
                    tokens_pop.append(token._.lemma)
                    sentence_start = span[0].sent.start
                    sentence_end = span[-1].sent.end
                    has_pop.append(doc[sentence_start : sentence_end].text)

        c_volk = Counter(([token._.is_volk for token in doc]))
        c_neg_elite = Counter(([token._.is_neg_elite for token in doc]))
        tokens_pop_counter = Counter(tokens_pop)

        if has_pop:
            res_dict['pop'] = True
        res_dict['doc'] = label
        res_dict['sents'] = has_pop
        res_dict['elite'] = c_neg_elite[True]
        res_dict['volk'] = c_volk[True]
        res_dict['len'] = len(doc)
        res_dict['volk_'] = hits['volk']
        res_dict['elite_'] = hits['elite']
        res_dict['volk_counter'] = Counter(hits['volk'])
        res_dict['elite_counter'] = Counter(hits['elite'])
        res_dict['hits'] = found
        res_dict['lemma_pop'] = tokens_pop_counter
        res.append(res_dict)
