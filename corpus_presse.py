# %%
import pandas as pd
import glob

# %%
PATH = '/media/philippy/SSD/data/ma/presse/'
files_csv = glob.glob(PATH + '*')

# %%
df2 = pd.read_csv('data/clean/mdbs_metadata_200802.csv')
df_from_each_file = (pd.read_csv(f, sep='|') for f in files_csv)
df   = pd.concat(df_from_each_file, ignore_index=True)

# %%
df2.name_res = df2.name_res.str.lower()
df.name_res = df.name_res.str.lower()

# %%
# fill id_party na with CDU/CSU
df.id_party.fillna(2.0, inplace=True)

# %%
# replace nas with party
map = ['AfD Partei', 'CDU Partei', 'CSU Partei', 'DIE GRÜNEN Partei', 'DIE LINKE Partei', 'Die blaue Partei', 'FDP Partei', 'SPD Partei']
for i in df.id_party.unique():
    print(i)
    df.loc[df.id_party == i, 'name_res'] = df[df.id_party == i].name_res.fillna(map[int(i-1)])

# %%
# check for fitting name_res
[i for i in df.name_res.unique() if i.lower() not in df2.name_res.unique()]

# schockenhoff/golze/becker/tiefensee/sieling/hinz/strobl: hinzufügen

# %%
# create corpus files:
# set id for each speech and write textfiles to disc

df['join'] = df[['presse_header', 'presse_text']].fillna('').apply(lambda x: ' '.join(x), axis=1)

# %%
# drop old metacollumns from df
df.drop(columns=['party', 'id_mdb', 'id_party', 'agw_18', 'agw_19',
       'birth_year', 'education', 'election_list', 'gender', 'last_name',
       'first_name', 'social_media_profile', 'aw_profil_url', 'identifier',
       'profiles_count', 'facebook', 'twitter', 'youtube', 'instagram',
       'flickr', 'is_add', 'last_name_merge'], inplace=True)

# %%
# merge with metadata
df = df.merge(df2, how='left', on='name_res')

# %%
filenames = []
for i, text in enumerate(df['join'].to_list()):
    filename = 'presse_{:0>6}.txt'.format(i)
    filenames.append(filename)
    with open('/media/philippy/SSD/data/ma/corpus/presse/{}'.format(filename), "w", encoding='utf8', newline='\n') as text_file:
        text_file.write(text)

df['file_id'] = filenames
df['filename'] = filenames
print(len(filenames))

# %%
# save metadata as JSON
df = df.set_index('filename')
df.loc[:, [i for i in df.columns if i not in ['presse_text', 'presse_header', 'join']]].to_json('presse_meta.json', orient='index')

# %%

# %%
# fill na with party
# m = df.id_party == 1.0
# df.loc[m, 'name_res'] = df[m].name_res.fillna('AfD Partei')
# m = df.id_party == 2.0
# df[m].name_res.fillna('CDU Partei', inplace=True)
# m = df.id_party == 3.0
# df[m].name_res.fillna('CSU Partei', inplace=True)
# m = df.id_party == 4.0
# df[m].name_res.fillna('DIE GRÜNEN Partei', inplace=True)
# m = df.id_party == 5.0
# df[m].name_res.fillna('DIE LINKE Partei', inplace=True)
# m = df.id_party == 6.0
# df[m].name_res.fillna('Die blaue Partei', inplace=True)
# m = df.id_party == 7.0
# df[m].name_res.fillna('FDP Partei', inplace=True)
# m = df.id_party == 8.0
# df[m].name_res.fillna('SPD Partei', inplace=True)
# df.loc[:, 'id_party' == 1.0]
