# %%
import sys
import pandas as pd
import numpy as np
import requests
import json
from src.d00_utils import check

# %%
query = """{
  category(id: "7ee81691-0363-4cb5-bea8-aa43350ac1ac") {
    name
    isRoot
    ancestry {
      name
      id
      isRoot
      ancestryRoot{
        name
      }
    }
    organisationsInCategory: organisationCategoriesCount
    organisations{
      edges{
        node{
          name
          id
          profilesCount
          profiles(first: 20) {
            edges {
            node {
            urlToExternalProfile
            identifier
            isMasterProfile
          rankValueAttributeName
        }
      }
    }
        }
      }
    }
  }
}"""

# %%
response = requests.post(
    "https://pluragraph.de/api/graphql",
    headers={"authorization": "Bearer EHWfTjLAbvZtL4ASXYpyr7fh"},
    data={"query": query},
)

# %%
plural = json.loads(response.text)

with open("mdbs_pluragraph_data_200731.json", "w") as f:
    json.dump(plural, f, indent=4, sort_keys=True)

# %%
def extract_values(obj, key):
    """Pull all values of specified key from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    results = extract(obj, arr, key)
    return results

# %%
names = extract_values(plural, "urlToExternalProfile")
print(len(names))
names

# %%
politiker = []


pol = plural["data"]["category"]["organisations"]["edges"]
for data in pol:
    res = dict()
    res = {
        "name": "",
        "identifier": [],
        "profiles_count": 0,
        "profile": ["NaN"],
        "facebook": [],
        "twitter": [],
        "youtube": [],
        "instagram": [],
        "flickr": [],
    }
    data = data["node"]
    # print(data)
    res["name"] = data["name"]
    res["profiles_count"] = data["profilesCount"]
    res["identifier"] = [i["node"]["identifier"] for i in data["profiles"]["edges"]]

    profile = extract_values(data, "urlToExternalProfile")
    profile = [i for i in profile if i != None]
    res["profile"] = profile
    for i in profile:
        # print(i)
        if i.startswith("https://www.facebook") | i.startswith("https://facebook"):
            try:
                res["facebook"].append(i)
            except KeyError:
                res["facebook"] = i

        if i.startswith("https://www.twitter") | i.startswith("https://twitter"):
            try:
                res["twitter"].append(i)
            except KeyError:
                res["twitter"] = i
        if i.startswith("https://www.instagram") | i.startswith("https://instagram"):
            try:
                res["instagram"].append(i)
            except KeyError:
                res["instagram"] = i
        if i.startswith("https://www.youtube") | i.startswith("https://youtube"):
            try:
                res["youtube"].append(i)
            except KeyError:
                res["youtube"] = i
        if i.startswith("https://secure.flickr") | i.startswith("https://flickr"):
            try:
                res["flickr"].append(i)
            except KeyError:
                res["flickr"] = i

    politiker.append(res)

politiker

pd.set_option("display.max_columns", 200)

# %%
df = pd.DataFrame(politiker)

# with pd.option_context("display.max_rows", None, "max_colwidth", -1):
    # display(df)

# %%
df.to_csv('mdbs_plura_200731.csv', mode='w', index=False, sep=',', header=True)

# %%
# plura from disk:
df = pd.read_csv('data/raw/mdbs/mdbs_plura_200731.csv')

# %%
# merge metadata: pluragraph & abgeordnetenwatch

# load wp 18 data
df18 = pd.read_csv('data/raw/mdbs/mdbs_aw_wp18.csv')
# df18 = pd.read_csv('projects/content_analysis/data/raw/mdbs/mdbs_aw_wp18.csv')
df18['name'] = df18[['first_name', 'last_name']].apply(lambda x: ' '.join(x), axis=1)
df18['WP18'] = True

# %%
# load wp19 data
df19 = pd.read_csv('data/raw/mdbs/mdbs_aw_wp19.csv')
df19['profile_url'] = df19['meta/url']
df19['first_name'] = df19['personal/first_name']
df19['last_name'] = df19['personal/last_name']
df19['gender'] = df19['personal/gender']
df19['birthyear'] = df19['personal/birthyear']
df19['education'] = df19['personal/education']
df19['picture'] = df19['personal/picture/url']
df19['party'] = df19['party']
df19['election_list'] = df19['list/name']
df19['list_won'] = df19['list/won']
df19['agw_id'] = df19['meta/uuid']
df19['name'] = df19[['first_name', 'last_name']].apply(
    lambda x: ' '.join(x), axis=1)
df19['WP19'] = True

# %%
# merge wp 18 & 19
df_merge = pd.merge(df18, df19, on="name", how='outer', suffixes=('_18','_19'))
df_merge.birth_date = df_merge['birth_date'].apply(lambda x: str(x)[:4])

# %%
# set column names, remove nans, create first / last name
df_all = pd.DataFrame()
df_all['name'] = df_merge['name']
df_all['agw_18'] = df_merge['agw_id_18']
df_all['agw_19'] = df_merge['agw_id_19']
df_all['party'] = np.where(df_merge['party_19'] != 'NaN', df_merge['party_18'], df_merge['party_19'])
df_all['profile_url'] = np.where(df_merge['profile_url_19'] != 'NaN', df_merge['profile_url_18'], df_merge['profile_url_19'])
df_all['birth_year'] = np.where(df_merge['birthyear'] != 'NaN', df_merge['birth_date'], df_merge['birthyear'])
labels = ['first_name', 'last_name', 'gender', 'education', 'election_list', ]
for i in labels:
    df_all['{}'.format(i)] = np.where(df_merge['{}_19'.format(i)] == 'NaN', df_merge['{}_19'.format(i)], df_merge['{}_18'.format(i)])
labels = ['party', 'first_name', 'last_name', 'gender', 'education', 'election_list']
df_all['profile_url'] = np.where(df_all['profile_url'] != 'NaN', df_merge['profile_url_19'], df_merge['profile_url_18'])
df_all['birth_year'] = np.where(df_all['birth_year'] != 'NaN', df_merge['birthyear'], df_merge['birth_date'])
for i in labels:
    df_all[i].fillna(df_merge['{}_19'.format(i)], inplace=True)
df_all.birth_year.fillna(df_merge.birth_date, inplace=True)
df_all.profile_url.fillna(df_merge.profile_url_18, inplace=True)
df_all['name'] = df_all.apply(lambda row: row.first_name + ' ' + row.last_name, axis=1)
df = df_all
df['name_clean'] = df.apply(lambda row: row['name'].strip(), axis=1)
df['name_clean_'] = df['name_clean']

# %%
# load plugagraph data
df2 = pd.read_csv('data/raw/mdbs/mdbs_plura_200731.csv', sep=',')

# %%
# remove titles for names in pluragraph df2
df2['name_clean'] = df2.apply(lambda row: row['name'].split('Dr.')[-1], axis=1)
df2['name_clean'] = df2.apply(lambda row: row['name_clean'].split('Prof.')[-1], axis=1)
df2['name_clean'] = df2.apply(lambda row: row['name_clean'].split('Dr.-Ing.')[-1], axis=1)
df2['name_clean'] = df2.apply(lambda row: row['name_clean'].split('h.c.')[-1], axis=1)
df2['name_clean'] = df2.apply(lambda row: row['name_clean'].split('h. c.')[-1], axis=1)
df2['name_clean'] = df2.apply(lambda row: row['name_clean'].split('med.')[-1], axis=1)
df2['name_clean'] = df2.apply(lambda row: row['name_clean'].split('med')[-1], axis=1)
df2['name_clean'] = df2.apply(lambda row: row['name_clean'].split('-Ing.')[-1], axis=1)
df2['name_clean'] = df2.apply(lambda row: row['name_clean'].strip(), axis=1)
df2['name_clean_'] = df2['name_clean']

# %%
# fix names in aw df
new = {'Konstantin Elias Kuhle': 'Konstantin Kuhle', 'Fabio De Masi': 'Fabio de Masi', 'Jan R. Nolte': 'Jan Nolte', 'Helin Evrim Sommer': 'Evrim Sommer', 'Eva-Maria Schreiber': 'Eva Schreiber', 'Reinhard Arnold Houben': 'Reinhard Houben', 'Ulrich Oehme': 'Ulrich Öhme', 'Tobias Matthias Peterka': 'Tobias Peterka', 'Siegbert Droese': 'Siegbert Dröse'}
df['name_res'] = df['name_clean']
for k,v in new.items():
    df['name_res'].loc[(df['name_clean_'] == k)] = v

# %%
# fix names in pluragraph df2
new = {'Matthias Büttner (Stendal)': 'Matthias Büttner', 'Sevim Dağdelen': 'Sevim Dagdelen', 'Aydan Özoğuz': 'Aydan Özoguz', 'Konstantin Elias Kuhle': 'Konstantin Kuhle', 'Fabio De Masi': 'Fabio de Masi', 'Diether Dehm-Desoi': 'Diether Dehm', 'Jan R. Nolte': 'Jan Nolte', 'Erwin Josef Rüddel': 'Erwin Rüddel', 'Helin Evrim Sommer': 'Evrim Sommer', 'Frank Michael Junge': 'Frank Junge', 'Eva-Maria Schreiber': 'Eva Schreiber', 'Christoph J. Ploß': 'Christoph Ploß', 'Detlef Müller (Sachsen)': 'Detlef Müller', 'Reinhard Arnold Houben': 'Reinhard Houben', 'Ulrich Oehme': 'Ulrich Öhme', 'Michael Peter Groß': 'Michael Groß', 'Tobias Matthias Peterka': 'Tobias Peterka', 'Paul V. Podolay': 'Paul Podolay', 'Michael Georg Link': 'Michael Link', 'Heiko Hessenkemper': 'Heiko Heßenkemper', 'Jörg Cézanne': 'Jörg Cezanne', 'Siegbert Droese': 'Siegbert Dröse', 'Uwe Schmidt (Bremen)': 'Uwe Schmidt', 'Andreas G. Lämmel': 'Andreas Lämmel'}
df2['name_res'] = df2['name_clean']
for k,v in new.items():
    df2['name_res'].loc[(df2['name_clean_'] == k)] = v

# %%
# fix missing or misspelled names
df2.loc[df2.name_res == 'Ali', 'name_res'] = 'Amira Mohamed Ali'

# %% check for missing names in aw data, only [] should be left
mask = df2.name_res.isin(df.name_res)
df2[~mask]

# %% final merge
df_res = pd.merge(df, df2, left_on='name_res', right_on='name_res',
                  how='outer', suffixes=('_left', '_right'))
df_res.drop(columns=['name_clean_left', 'name_clean__left', 'name_right', 'name_left', 'name_clean_right', 'name_clean__right'], inplace=True)

# %%
# fix formatting of links
cols = ['identifier', 'profile', 'facebook', 'twitter', 'youtube', 'instagram', 'flickr']
df_res.loc[:, cols] = df_res[cols].apply(lambda x: x.str.strip(']'))
df_res.loc[:, cols] = df_res[cols].apply(lambda x: x.str.strip('['))
df_res.loc[:, cols] = df_res[cols].apply(lambda x: x.str.replace("'", ''))


# %%
# add missing members
df_add = pd.DataFrame({'name_res': ['Hans-Peter Bartels'], 'facebook': ['https://www.facebook.com/Hans-Peter-Bartels'], 'twitter': ['https://twitter.com/hanspbartels']})

df_res = df_res.append(df_add, ignore_index=True)

# add missing from presse
df_add = pd.DataFrame({'name_res': ['Thomas Strobl', 'Andreas Schockenhoff', 'Diana Golze', 'Dirk Becker', 'Wolfgang Tiefensee', 'Carsten Sieling', 'Petra Hinz', 'Philipp Mißfelder'], 'party': ['SPD', 'CDU', 'DIE LINKE', 'SPD', 'SPD', 'SPD', 'SPD', 'CDU'], 'gender': ['male', 'male', 'female', 'male', 'male', 'male', 'female', 'male']})
df_res = df_res.append(df_add, ignore_index=True)

# ad missing from twitter
# df_add = pd.DataFrame({'name_res': ['Beate Walter-Rosenheimer', 'Dagmar Schmidt', 'Karl-Heinz Brunner', 'Johannes Huber', 'Gerold Otten', 'Sylvia Kotting-Uhl'], 'party' : ['DIE GRÜNEN', 'SPD', 'SPD', 'AfD', 'AfD', 'DIE GRÜNEN'], 'gender': ['female', 'female', 'male', 'male', 'male', 'female'], 'twitter': ['https://twitter.com/beatewaro', 'https://twitter.com/dieschmidt', 'https://twitter.com/brunnerganzohr', 'https://twitter.com/hubermdb', 'https://twitter.com/gerold_otten', 'https://twitter.com/kottinguhl']})

# replace twitter-profiles:


# %%
# add missing data (party-membership)
df_res.loc[df_res.name_res == 'Dorothee Martin', ['party', 'gender']] = ['SPD', 'female']
df_res.loc[df_res.name_res == 'Saskia Ludwig', ['party', 'gender']] = ['CDU', 'female']
df_res.loc[df_res.name_res == 'Bela Bach', ['party', 'gender']] = ['SPD', 'female']
df_res.loc[df_res.name_res == 'Sandra Bubendorfer-Licht', ['party', 'gender']] = ['FDP', 'female']
df_res.loc[df_res.name_res == 'Sylvia Lehmann', ['party', 'gender']] = ['SPD', 'female']
df_res.loc[df_res.name_res == 'Reginald Hanke', ['party', 'gender']] = ['FDP', 'male']
df_res.loc[df_res.name_res == 'Hans-Peter Bartels', ['party', 'gender']] = ['SPD', 'male']

# from presse:
# df_res.loc[df_res.name_res == 'Andreas Schockenhoff', ['party', 'gender']] = ['CDU', 'male']



# %%
# add cases for party-profiles
df_add = pd.DataFrame({'name_res': ['AfD Partei', 'CDU Partei', 'CSU Partei', 'DIE GRÜNEN Partei', 'DIE LINKE Partei', 'Die blaue Partei', 'FDP Partei', 'SPD Partei'], 'facebook': ['https://www.facebook.com/afdimbundestag/', 'https://www.facebook.com/cducsubundestagsfraktion/', 'https://www.facebook.com/cducsubundestagsfraktion/', 'https://www.facebook.com/Gruene.im.Bundestag', 'https://www.facebook.com/linksfraktion', 'https://www.facebook.com/DieBlauen/', 'https://www.facebook.com/fdpbt/', 'https://www.facebook.com/spdbundestagsfraktion/'], 'twitter': ['https://twitter.com/afdimbundestag', 'https://twitter.com/cducsubt', 'https://twitter.com/cducsubt',  'https://twitter.com/gruenebundestag', 'https://twitter.com/linksfraktion', 'https://twitter.com/BlaueFraktion', 'https://twitter.com/fdpbt', 'https://twitter.com/spdbt'], 'youtube':['https://www.youtube.com/channel/UC_dZp8bZipnjntBGLVHm6rw', 'https://www.youtube.com/channel/UCWpop4RlpejOFLebR0C1YaQ', 'https://www.youtube.com/channel/UCWpop4RlpejOFLebR0C1YaQ', 'https://www.youtube.com/channel/UC7TAA2WYlPfb6eDJCeX4u0w', 'https://www.youtube.com/channel/UCF2SPLBq18sL88yZw9m-GZQ', None, 'https://www.youtube.com/channel/UC2TCluB4jcrJqis5rzGcCbw', 'https://www.youtube.com/channel/UCUVSxH8r5fj3Uki5uAcyQaw'], 'party':['AfD', 'CDU', 'CSU', 'DIE GRÜNEN', 'DIE LINKE', 'Die blaue Partei', 'FDP', 'SPD']})
df_res = df_res.append(df_add, ignore_index=True)

# %%
# add ids for mdbs and parties
df_res.loc[df_res['party'] == 'fraktionslos', 'party'] = 'Parteilos'
df_res.loc[df_res['party'] == 'Parteilos']

df_res['id_party'] = 99
df_res['id_mdb'] = np.nan

df_res.loc[df_res['party'] == 'Parteilos', 'id_party'] = 0
df_res.loc[df_res['party'] == 'AfD', 'id_party'] = 1
df_res.loc[df_res['party'] == 'CDU', 'id_party'] = 2
df_res.loc[df_res['party'] == 'CSU', 'id_party'] = 3
df_res.loc[df_res['party'] == 'DIE GRÜNEN', 'id_party'] = 4
df_res.loc[df_res['party'] == 'DIE LINKE', 'id_party'] = 5
df_res.loc[df_res['party'] == 'Die blaue Partei', 'id_party'] = 6
df_res.loc[df_res['party'] == 'FDP', 'id_party'] = 7
df_res.loc[df_res['party'] == 'SPD', 'id_party'] = 8

for i, name in enumerate(df_res['name_res'].unique()):
    df_res.loc[df_res['name_res'] == name, 'id_mdb'] = df_res.loc[df_res['name_res']
                                                      == name].id_party.to_string(index=False) + str(i+1).zfill(3)

# %%
# fix party-profiles id
p = ['AfD Partei', 'CDU Partei', 'CSU Partei', 'DIE GRÜNEN Partei', 'DIE LINKE Partei', 'Die blaue Partei', 'FDP Partei', 'SPD Partei']

ids = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
for i, j in zip(p, ids):
    df_res.loc[df_res['name_res'] == i, 'id_mdb'] = j

# %%
# names to lowercase
df_res.name_res = df_res.name_res.str.lower()

# %%
# save dataframe as csv
df_res.to_csv('mdbs_metadata_200802.csv', mode='w', index=False, sep=',', header=True)


# %%
