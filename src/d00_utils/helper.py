#%%
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def flatten(arr):
    """Flatten array."""
    for i in arr:
        if isinstance(i, list):
            yield from flatten(i)
        elif isinstance(i, set):
            yield from flatten(i)
        else:
            yield i

def fix_umlauts(old_string):
    table_umlauts = {"ÃŸ": "ß", "ãÿ": "ß", "ã¤": "ä", "ã¼": "ü", "ã¶": "ö", 'Ã„': 'Ä', "Ãœ": "Ü", "Ã–": "Ö", 'â‚¬': '€'}
    for v, k in table_umlauts.items():
        old_string = old_string.replace(v, k)
    print(old_string)
    return old_string

def emb_fix_umlauts(emb):
    new_dict = {}
    for key in emb.wv.key_to_index:
        # print(key)
        old_key = key
        new_key = fix_umlauts(key)
        new_dict[new_key] = emb.wv.key_to_index[old_key]
    emb.wv.key_to_index = new_dict

    for i, key in enumerate(emb.wv.index_to_key):
        emb.wv.index_to_key[i] = fix_umlauts(key)
    return emb

def filter_spans_overlap(spans):
    """Filter a sequence of spans and remove duplicates AND DIVIDE!!! overlaps. Useful for
    creating named entities (where one token can only be part of one entity) or
    when merging spans with `Retokenizer.merge`. When spans overlap, the (first)
    longest span is preferred over shorter spans.

    spans (iterable): The spans to filter.
    RETURNS (list): The filtered spans.
    """
    # get_sort_key = lambda span: (span['span_end'] - span['span_start'], -span['span_start'])
    # sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    sorted_spans = sorted(spans, key=lambda span: span['span_start'])
    # print(sorted_spans)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        # current_start = span['span_start']
        # current_end = span['span_end']
        # current_stop =
        # Check for end - 1 here because boundaries are inclusive
        if span['span_start'] in seen_tokens:
            for s in result:
                if s['span_start'] == result[-1]['span_start']:
                    if s['span_end'] < span['span_end']:
                        s['span_end'] = span['span_end']
                    else:
                        span['span_end'] = s['span_end']

            span['span_start'] = result[-1]['span_start']
            result.append(span)

        elif span['span_start'] not in seen_tokens and span['span_end']- 1 not in seen_tokens:
            result.append(span)


        seen_tokens.update(range(span['span_start'], span['span_end']))
    result = sorted(result, key=lambda span: span['span_start'])
    return result


# example = [{'span_start': 0, 'span_end': 3}, {'span_start': 2, 'span_end': 8}, {'span_start': 200, 'span_end': 250}, {'span_start': 220, 'span_end': 300}]
# d = filter_spans_overlap(example)
# d

def filter_spans_overlap_no_merge(spans):
    sorted_spans = sorted(spans, key=lambda span: span['span_start'])
    result = []
    seen_tokens = set()
    seen_starts = dict()
    for span in sorted_spans:
        span_len = span['span_end'] - span['span_start']
        # Check for end - 1 here because boundaries are inclusive
        if span['span_start'] not in seen_starts and span['span_end'] not in seen_tokens:
            seen_starts[span['span_start']] = span_len
            result.append(span)

        elif span['span_start'] in seen_starts:
            if span['span_end']-1 not in seen_tokens:
                if span_len > seen_starts[span['span_start']]:
                    seen_starts[span['span_start']] = span_len
                    for r in result:
                        if r['span_start'] == span['span_start']:
                            r['span_end'] = span['span_end']


                # if s['span_start'] == result[-1]['span_start']:
                #     if s['span_end'] < span['span_end']:
                #         s['span_end'] = span['span_end']
                #     else:
                #         span['span_end'] = s['span_end']

            # span['span_start'] = result[-1]['span_start']
            # result.append(span)

        elif span['span_start'] not in seen_tokens and span['span_end']- 1 not in seen_tokens:
            result.append(span)

        seen_tokens.update(range(span['span_start'], span['span_end']))
    result = sorted(result, key=lambda span: span['span_start'])
    return result

# example = [{'span_start': 0, 'span_end': 3}, {'span_start': 2, 'span_end': 8}, {'span_start': 200, 'span_end': 250}, {'span_start': 220, 'span_end': 300}, {'span_start': 0, 'span_end': 6}, {'span_start': 1, 'span_end': 220}]
# d = filter_spans_overlap_no_merge(example)
# d

# %%
