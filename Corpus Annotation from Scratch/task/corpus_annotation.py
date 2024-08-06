import en_core_web_sm
import numpy as np
import re
import pandas as pd
from nltk.corpus import words
from scipy.stats import pearsonr
from collections import Counter

def main():
    data = np.array(['Token', 'Lemma', 'POS', 'Entity_type', 'IOB_tag', 'Lang', 'Non_English', 'Shape'])
    en_sm_model = en_core_web_sm.load()
    words_set = set(words.words())

    def is_non_english(token):
        return len(token.lemma_) > 4 and token.pos_ in ['ADJ', 'ADV', 'NOUN', 'VERB'] and token.lemma_ not in words_set

    # file = open('./test/clockwork_orange.txt', 'r')
    file = open(input(), 'r')
    doc = en_sm_model(file.read())

    for token in doc:
        if not (token.like_url or token.is_title or token.is_space or token.is_punct or re.search(r'[><_\\/*]', token.text)):
            data = np.vstack([data, [token.text, token.lemma_, token.pos_, token.ent_type_, token.ent_iob_, token.lang_, is_non_english(token), token.shape_]])

    df = pd.DataFrame(data[1:], columns=data[0])
    df_ents = pd.Series(doc.ents)

    print(f'Number of multi-word named entities: {df_ents[df_ents.astype(str).str.contains(" ")].count()}')
    print(f'Number of lemmas \'devotchka\': {df["Lemma"].str.contains("devotchka").sum()}')
    print(f'Number of tokens with the stem \'milk\': {df["Lemma"].str.contains("milk").sum()}')

    # ents = Counter()
    #
    # for ent in doc.ents:
    #     ents[f"{ent.label_}"] += 1

    print(f'Most frequent entity type: {Counter(ent.label_ for ent in doc.ents if ent.label_).most_common(1)[0][0]}')
    print(f'Most frequent named entity token: {tuple([df_ents.astype(str).value_counts().keys().tolist()[0], df_ents.value_counts().keys().tolist()[0].label_])}')
    print(f'Most common non-English words: {dict(df[df.Non_English == "True"]["Lemma"].value_counts().head(10))}')

    def replace_func(x):
        if x in ['NOUN', 'PROPN']:
            return 1
        else:
            return 0

    df['POS'] = df['POS'].apply(replace_func)
    df['Entity_type'] = df['Entity_type'].astype(str).apply(lambda x: 1 if True else 0)

    print(f'Correlation between NOUN and PROPN and named entities: {round(pearsonr(df["POS"], df["Entity_type"])[0], 2)}')

if __name__ == "__main__":
    main()

