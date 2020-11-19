# %%
import pandas as pd
import glob
import re, string

# %%
PATH = '/media/philippy/SSD/data/ma/twitter/'
files = glob.glob(PATH + '*')


# %%
df2 = pd.read_csv('data/clean/mdbs_metadata_200802.csv')

# %%
df_from_each_file = (pd.read_json(f) for f in files)
df = pd.concat(df_from_each_file, ignore_index=True)

# %%
# remove URLS
def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)
df['text_clean'] = df['text'].apply(lambda x: remove_URL(x))


# %%
# df.set_index('datetime')

grouper = df.set_index('timestamp').groupby([pd.Grouper(freq='1M'), 'screen_name'])
res = grouper.agg({'text_clean': ' - '.join}).reset_index()
res

# %%
# check if screen names in metadata
df2['screen_name'] = df2['twitter'].fillna('').apply(lambda x: x.split('/')[-1])

# %%
df2.screen_name = df2.screen_name.str.lower()
df.screen_name = df.screen_name.str.lower()
res.screen_name = res.screen_name.str.lower()

[i for i in res.screen_name.unique() if i not in df2.screen_name.unique()]

# %%
# add by hand to correct name_res
res.loc[res.screen_name == 'beatewaro', 'screen_name'] = 'teambeate'
res.loc[res.screen_name == 'dieschmidt', 'screen_name'] = 'teamdieschmidt'
res.loc[res.screen_name == 'kaiser_spd', 'screen_name'] = 'lier_e'
res.loc[res.screen_name == 'schwarz_spd', 'screen_name'] = 'schwarz_mdb'
res.loc[res.screen_name == 'brunnerganzohr', 'screen_name'] = 'brunnerkarl'
res.loc[res.screen_name == 'schwarz_afd', 'screen_name'] = 'schwarzmdb'
res.loc[res.screen_name == 'ulrich_oehme', 'screen_name'] = 'oehmeulrich'
res.loc[res.screen_name == 'gerold_otten', 'screen_name'] = 'ttte94'
res.loc[res.screen_name == 'fjunge', 'screen_name'] = 'frankjunge'
res.loc[res.screen_name == 'kottinguhl', 'screen_name'] = 'babetteschefin'
res.loc[res.screen_name == 'mdb_mueller_afd', 'screen_name'] = 'mueller_mdb'
res.loc[res.screen_name == 'waldemarherdt', 'screen_name'] = 'holdingeuropa'
res.loc[res.screen_name == 'hubermdb', 'screen_name'] = 'huber_afd'

# len(res[res.screen_name == 'teamdieschmidt'])
# %%
[i for i in res.screen_name.unique() if i not in df2.screen_name.unique()]

# %%
# merge with metadata
df_res = res.merge(df2, how='inner', on='screen_name')

# %%

filenames = []
for i, text in enumerate(df_res['text_clean'].to_list()):
    filename = 'twitter_{:0>6}.txt'.format(i)
    filenames.append(filename)
    with open('/media/philippy/SSD/data/ma/corpus/twitter/{}'.format(filename), "w", encoding='utf8', newline='\n') as text_file:
        text_file.write(text)

df_res['file_id'] = filenames
df_res['filename'] = filenames

print(len(filenames))
# %%
# save metadata as JSON
df_res = df_res.set_index('filename')
df_res.loc[:, [i for i in df_res.columns if i not in ['text_clean']]].to_json('twitter_meta.json', orient='index')

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
