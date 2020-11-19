# %%
from pprint import pprint
import json
import pandas as pd
import glob
from bs4 import BeautifulSoup

# %%


def parse_xml(file):
    '''parse xml-files of plenary protocols of WP18 (2013-2017) for date and protocol-number'''
    infile = open(file, "r", encoding='utf-8')
    contents = infile.read()
    soup = BeautifulSoup(contents, 'xml')

    datum = soup.find('DATUM')
    datum = datum.text

    nummer = soup.find('NR')
    nummer = nummer.text
    return datum, nummer


# combine text from csv + date & number from xml
files_csv = glob.glob('data/raw/plenar/WP18/*.csv')
files_xml = glob.glob('data/raw/plenar/WP18/WP18_xml/*.xml')

dfs = []
for i, (file_csv, file_xml) in enumerate(zip(files_csv, files_xml)):
    df = pd.read_csv(file_csv)
    datum, nummer = parse_xml(file_xml)
    df['datum'] = datum
    df['nummer'] = nummer
    df['filenum'] = str(file_xml)
    dfs.append(df)

df = pd.concat(dfs)


# %%
# filter plenary speeches (exclude questions, zwischenrufe, etc.) and combine to list
df_filter = df.loc[df['type'] == 'speech', :]
df_text = df_filter.groupby(['speaker_key', 'top_id'], as_index=False)[
    'text'].apply(list)
df_wp18 = pd.merge(pd.DataFrame(df_text), df_filter, on=[
                   'speaker_key', 'top_id'], how='inner')

# speeches as contineous string
df_wp18['text_clean'] = (
    df_wp18[0]
    .map(lambda x: [str(y) for y in x])
    .map(lambda x: ' '.join(x))
)

df_wp18.drop_duplicates(subset='text_clean', keep="first", inplace=True)

# check number of speeches of mdb Volker Beck, should be 137 according to offenesparlament.de
df_wp18.loc[df_wp18['speaker_fp'] == 'volker-beck', :].shape

# df_wp18.to_csv('../../data/raw/plenar/plenar_WP18.csv', mode='w', index=False, sep='|')


# %%
df_wp18.loc[df_wp18['datum'].isna()]


# %%
# drop unnecessary columns, convert datum to datetime
df_wp18.drop(labels=['Unnamed: 0', 'text', 0], axis=1, inplace=True)
df_wp18['datum'] = pd.to_datetime(df_wp18['datum'])
df_wp18.sort_values(by='datum').iloc[[0, -1]]


# %%
def parse_xml(file):
    print(file)
    infile = open(file, "r", encoding='utf-8')
    contents = infile.read()
    soup = BeautifulSoup(contents, 'lxml')
    reden = soup.find_all('rede')
    daten = soup.find('veranstaltungsdaten')
    datum = daten.find('datum')
    date = datum['date']

    plenar_file = []
    for rede in reden:
        # plenar_file.append(rede)

        plenar = dict()
        plenar['rede_id'] = rede['id']
        # print(plenar['rede_id'])
        vorname = rede.find('vorname')
        nachname = rede.find('nachname')
        namenszusatz = rede.find('namenszusatz')

        if vorname == None:
            vorname = 'NA'
        else:
            vorname = vorname.text

        if nachname == None:
            nachname = 'NA'
        else:
            nachname = nachname.text

        if namenszusatz != None:
            plenar['name'] = str(str(vorname) + ' ' +
                                 str(namenszusatz.text) + ' ' + str(nachname))
        else:
            plenar['name'] = str(str(vorname) + ' ' + str(nachname))

        rolle = rede.find('rolle_lang')
        if rolle != None:
            plenar['rolle'] = rolle.text
        else:
            plenar['rolle'] = 'NA'

        fraktion = rede.find('fraktion')
        if fraktion != None:
            plenar['fraktion'] = fraktion.text
        else:
            plenar['fraktion'] = 'NA'

        id_redner = rede.find('redner')
        plenar['id_redner'] = id_redner['id']
        plenar['id_rede'] = rede['id']
        plenar['date'] = date
        plenar['filenum'] = str(file)

        # Clear: Kommentare, Namen und Text von (Vize-)Präsidenten
        kommentare = rede.find_all('kommentar')
        for kommentar in kommentare:
            kommentar.clear()

        rede_res = []

        name_edits = rede.find_all('name')
        for name_edit in name_edits:
            next_p = name_edit.find_next_sibling("p")

            while True:
                if next_p == None:
                    break
                try:
                    if next_p['klasse'] == 'redner':
                        check_redner = next_p.find('redner')
                        # print(check_redner)
                        if check_redner['id'] == plenar['id_redner']:
                            break
                        else:
                            next_p = next_p.find_next_sibling('p')
                except:
                    break

                else:
                    delete_p = next_p
                    next_p = next_p.find_next_sibling("p")
                    delete_p.clear()

            name_edit.clear()
        relevant = True

        for p in rede.find_all('p'):
            try:
                if p['klasse'] == 'redner':
                    red = p.find('redner')
                    if red['id'] != plenar['id_redner']:
                        relevant = False
                    else:
                        relevant = True

                else:
                    if relevant:
                        rede_res.append(p.text)
                        # print(p.text)
            except:
                if relevant:
                    rede_res.append(p.text)

        plenar['text'] = " ".join(rede_res)
        plenar_file.append(plenar)

    return plenar_file


files = glob.glob('data/raw/plenar/WP19/*.xml')
# files = glob.glob('data/raw/plenar/WP19/19007-data.xml')

plenar_all = []
for file in files:
    plenar_all.extend(parse_xml(file))

df_wp19 = pd.DataFrame(plenar_all)
df_wp19.shape


# %%
# RENAME
df_wp19[['name', 'rolle']] = df_wp19[['name',  'rolle']].fillna('')
df_wp19['speaker_cleaned'] = df_wp19.apply(
    lambda x: x['name'].replace(x['rolle'], ''), axis=1)
df_wp19.rename(columns={'fraktion': 'speaker_party', 'date': 'datum',
                        'name': 'speaker', 'id_redner': 'speaker_key', 'text': 'text_clean'}, inplace=True)

df_wp19['wahlperiode'] = 19
df_wp19['datum'] = pd.to_datetime(df_wp19['datum'])
df_wp19.sort_values(by='datum')

df_wp19.head(1)


# %%
# concatenate wp18 & wp19
df_all = pd.concat([df_wp18, df_wp19], axis=0, ignore_index=True)
len(df_all)

# %%
df_all.dropna(subset=['text_clean'], inplace=True)
df_all.sort_values(by='datum', inplace=True)
len(df_all)

# %%
# - merge with metadata
df = pd.read_csv('data/mdbs_metadata.csv')
df_plenar = df_all


# %%
df_plenar['speaker_cleaned'] = df_plenar.apply(
    lambda row: row['speaker_cleaned'].split('Dr.')[-1], axis=1)
df_plenar['speaker_cleaned'] = df_plenar.apply(
    lambda row: row['speaker_cleaned'].split('Prof.')[-1], axis=1)
df_plenar['speaker_cleaned'] = df_plenar.apply(
    lambda row: row['speaker_cleaned'].split('Dr.-Ing.')[-1], axis=1)
df_plenar['speaker_cleaned'] = df_plenar.apply(
    lambda row: row['speaker_cleaned'].split('h.c.')[-1], axis=1)
df_plenar['speaker_cleaned'] = df_plenar.apply(
    lambda row: row['speaker_cleaned'].split('h. c.')[-1], axis=1)
df_plenar['speaker_cleaned'] = df_plenar.apply(
    lambda row: row['speaker_cleaned'].split('med.')[-1], axis=1)
df_plenar['speaker_cleaned'] = df_plenar.apply(
    lambda row: row['speaker_cleaned'].split('med')[-1], axis=1)
df_plenar['speaker_cleaned'] = df_plenar.apply(
    lambda row: row['speaker_cleaned'].split('-Ing.')[-1], axis=1)
df_plenar['speaker_cleaned'] = df_plenar.apply(
    lambda row: row['speaker_cleaned'].strip(), axis=1)
df_plenar['speaker_cleaned_'] = df_plenar['speaker_cleaned']


# %%
# to check
dfp = df_plenar

l1 = dfp.speaker_cleaned.unique()
l2 = df.name_res.unique()

x = [y for y in l1 if y not in l2]

# %%

dfp.loc[dfp['speaker_cleaned'] == 'Angela Merkel,',
        'speaker_cleaned'] = 'Angela Merkel'
dfp.loc[dfp['speaker_cleaned'] == 'Dagmar G. Wöhrl',
        'speaker_cleaned'] = 'Dagmar Wöhrl'
dfp.loc[dfp['speaker_cleaned'] == 'Andreas G. Lämmel',
        'speaker_cleaned'] = 'Andreas Lämmel'
dfp.loc[dfp['speaker_cleaned'] == 'Franz Josef Jung',
        'speaker_cleaned'] = 'Franz-Josef Jung'
dfp.loc[dfp['speaker_cleaned'] == 'Aydan Özoğuz',
        'speaker_cleaned'] = 'Aydan Özoguz'
dfp.loc[dfp['speaker_cleaned'] == 'Sevim Dağdelen',
        'speaker_cleaned'] = 'Sevim Dagdelen'
dfp.loc[dfp['speaker_cleaned'] == 'Jan Ralf Nolte',
        'speaker_cleaned'] = 'Jan Nolte'
dfp.loc[dfp['speaker_cleaned'] == 'Alexander Graf Graf Lambsdorff',
        'speaker_cleaned'] = 'Alexander Graf Lambsdorff'
dfp.loc[dfp['speaker_cleaned'] == 'Michael Georg Link',
        'speaker_cleaned'] = 'Michael Link'
dfp.loc[dfp['speaker_cleaned'] == 'Eberhardt Alexander Gauland',
        'speaker_cleaned'] = 'Alexander Gauland'
dfp.loc[dfp['speaker_cleaned'] == 'Fabio De Masi',
        'speaker_cleaned'] = 'Fabio de Masi'
dfp.loc[dfp['speaker_cleaned'] == 'Ulrich Oehme',
        'speaker_cleaned'] = 'Ulrich Öhme'
dfp.loc[dfp['speaker_cleaned'] == 'Armin-Paulus Hampel',
        'speaker_cleaned'] = 'Armin Paul Hampel'
dfp.loc[dfp['speaker_cleaned'] == 'Johann David Wadephul',
        'speaker_cleaned'] = 'Johann Wadephul'
dfp.loc[dfp['speaker_cleaned'] == 'Joana Eleonora Cotar',
        'speaker_cleaned'] = 'Joana Cotar'
dfp.loc[dfp['speaker_cleaned'] == 'Sonja Amalie Steffen',
        'speaker_cleaned'] = 'Sonja Steffen'
dfp.loc[dfp['speaker_cleaned'] == 'Konstantin Elias Kuhle',
        'speaker_cleaned'] = 'Konstantin Kuhle'
dfp.loc[dfp['speaker_cleaned'] == 'Roman Johannes Reusch',
        'speaker_cleaned'] = 'Roman Reusch'
dfp.loc[dfp['speaker_cleaned'] == 'Gero Clemens Hocker',
        'speaker_cleaned'] = 'Gero Hocker'
dfp.loc[dfp['speaker_cleaned'] == 'Ali',
        'speaker_cleaned'] = 'Amira Mohamed Ali'
dfp.loc[dfp['speaker_cleaned'] == 'Christian Freiherr von Freiherr Stetten',
        'speaker_cleaned'] = 'Christian Freiherr von Stetten'
dfp.loc[dfp['speaker_cleaned'] == 'Tobias Matthias Peterka',
        'speaker_cleaned'] = 'Tobias Peterka'
dfp.loc[dfp['speaker_cleaned'] == 'Mariana Iris Harder-Kühnel',
        'speaker_cleaned'] = 'Mariana Harder-Kühnel'
dfp.loc[dfp['speaker_cleaned'] == 'Johannes Graf Schraps',
        'speaker_cleaned'] = 'Johannes Schraps'
dfp.loc[dfp['speaker_cleaned'] == 'Siegbert Droese',
        'speaker_cleaned'] = 'Siegbert Dröse'
dfp.loc[dfp['speaker_cleaned'] == 'Martin Erwin Renner',
        'speaker_cleaned'] = 'Martin E. Renner'
dfp.loc[dfp['speaker_cleaned'] == 'Bettina Margarethe Wiesmann',
        'speaker_cleaned'] = 'Bettina Wiesmann '
dfp.loc[dfp['speaker_cleaned'] == 'Jan Ralf Graf Nolte',
        'speaker_cleaned'] = 'Jan Nolte'
dfp.loc[dfp['speaker_cleaned'] == 'Gerd Graf Müller',
        'speaker_cleaned'] = 'Gerd Müller'
dfp.loc[dfp['speaker_cleaned'] == 'Helin Evrim Sommer',
        'speaker_cleaned'] = 'Evrim Sommer'
dfp.loc[dfp['speaker_cleaned'] == 'Udo Theodor Hemmelgarn',
        'speaker_cleaned'] = 'Udo Hemmelgarn'
dfp.loc[dfp['speaker_cleaned'] == 'Eva-Maria Elisabeth Schreiber',
        'speaker_cleaned'] = 'Eva Schreiber'
dfp.loc[dfp['speaker_cleaned'] == 'Norbert Maria Altenkamp',
        'speaker_cleaned'] = 'Norbert Altenkamp'
dfp.loc[dfp['speaker_cleaned'] == 'Katharina Graf Dröge',
        'speaker_cleaned'] = 'Katharina Dröge'
dfp.loc[dfp['speaker_cleaned'] == 'Britta Katharina Dassler',
        'speaker_cleaned'] = 'Britta Dassler'
dfp.loc[dfp['speaker_cleaned'] == 'Michael Graf Leutert',
        'speaker_cleaned'] = 'Michael Leutert'
dfp.loc[dfp['speaker_cleaned'] == 'Eva-Maria Schreiber',
        'speaker_cleaned'] = 'Eva Schreiber'
dfp.loc[dfp['speaker_cleaned'] == 'Jens Graf Spahn',
        'speaker_cleaned'] = 'Jens Spahn'
dfp.loc[dfp['speaker_cleaned'] == 'Rolf Graf Mützenich',
        'speaker_cleaned'] = 'Rolf Mützenich'
dfp.loc[dfp['speaker_cleaned'] == 'Paul Viktor Podolay',
        'speaker_cleaned'] = 'Paul Podolay'
dfp.loc[dfp['speaker_cleaned'] == 'Martin Graf Hebner',
        'speaker_cleaned'] = 'Martin Hebner'
dfp.loc[dfp['speaker_cleaned'] == 'Albert H. Weiler',
        'speaker_cleaned'] = 'Albert Weiler'
dfp.loc[dfp['speaker_cleaned'] == 'Jens Graf Kestner',
        'speaker_cleaned'] = 'Jens Kestner'
dfp.loc[dfp['speaker_cleaned'] == 'Heidrun Bluhm-Förster',
        'speaker_cleaned'] = 'Heidrun Bluhm'
dfp.loc[dfp['speaker_cleaned'] == 'Elvan Korkmaz-Emre',
        'speaker_cleaned'] = 'Elvan Korkmaz'
dfp.loc[dfp['speaker_cleaned'] == 'Katharina Kloke',
        'speaker_cleaned'] = 'katharina willkomm'
dfp.loc[dfp['speaker_cleaned'] == 'in der beek',
        'speaker_cleaned'] = 'olaf in der beek'
#dfp.loc[dfp['speaker_cleaned'] == 'aaa'] = 'bbb'

# %%
dfp.speaker_cleaned = dfp.speaker_cleaned.apply(lambda x: x.lower().strip())

# %%
# check again
l1 = dfp.speaker_cleaned.unique()
l2 = df.name_res.unique()
remove = [y for y in l1 if y not in l2]

# %%
# df['is_add'] = 0
# df_add = pd.DataFrame({'name_res': ['Katharina Kloke', 'Hans-Peter Bartels'], 'is_add': [1, 1]})
# df = df.append(df_add, ignore_index = True)

dfp_drop = dfp.drop(dfp[dfp.speaker_cleaned.isin(remove)].index)


# %%
df_res = pd.merge(df, dfp_drop, left_on='name_res',
                  right_on='speaker_cleaned', how='inner', suffixes=('_left', '_right'))
df_res['typ'] = 'plenar'


# %%
df_res_rename = df_res.rename(columns={"filenum": "plenar_file", "id_rede": "plenar_id_rede", "wahlperiode": "plenar_wahlperiode",
                                       "profile_url_left": "aw_profil_url", "profile": "social_media_profile", "text_clean": "text"})


# %%
df_res_rename.columns


# %%
df_clean = df_res_rename[['name_res', 'party', 'id_mdb', 'id_party', 'agw_18', 'agw_19', 'birth_year', 'education', 'election_list', 'gender', 'last_name', 'first_name', 'social_media_profile',
                          'aw_profil_url', 'identifier', 'profiles_count', 'facebook', 'twitter', 'youtube', 'instagram', 'flickr', 'typ', 'datum', 'text', 'plenar_file', 'plenar_id_rede', 'plenar_wahlperiode']]


# %%
# set id for each speech and write textfiles to disc
# texts = [str(x) for x in df_clean['text']]
texts = df_clean['text'].tolist()

filenames = []
for i, text in enumerate(texts):
    filename = 'plenar_{:0>6}.txt'.format(i)
    filenames.append(filename)
    with open('data/corpus/plenar/{}'.format(filename), "w") as text_file:
        try:
            text_file.write(text)
        except:
            pass

df_clean['filename'] = filenames
df_clean['file_id'] = filenames
print(len(filenames))
print(df_clean.shape)


# %%
'''save metadata as JSON'''
df_clean = df_clean.set_index('filename')
df_clean.loc[:, df_clean.columns != 'text'].to_json(
    'data/plenar_meta.json', orient='index')

# pprint(data)
# print(data[]) "Dimension: ", data['cubes'][cube]['dim']


# %%
# check

json_file = 'plenar_meta.json'
with open(json_file) as json_data:
    data = json.load(json_data)
for i in range(10):
    pprint(data['plenar_00{}000.txt'.format(i)])
