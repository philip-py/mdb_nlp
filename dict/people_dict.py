# %%
#Volkszentrismus
#with sent / uni


attr_ordinary = [
    'einfach',
    'normal',
    'klein'
]

people = [
'allgemeinheit',
'arbeiter',
'arbeiterin',
'arbeitnehmer',
'arbeitnehmerin',
'beschäftigter',
'beschäftigte',
'bevölkerung',
'bürger',
'bürgerin',
'bürgerschaft',
'bürgertum',
'deutsche',
'gemeinwesen',
'gemeinwohl',
'gesamtbevölkerung',
'gesamtgesellschaft',
'gesellschaftsordnung',
'grundordnung',
'menschenverstand',
'mitbürger',
'landsleute',
'landsmann',
'landsfrau',
'normalbürger',
'normalbürgerin',
'öffentlichkeit',
'ottonormalverbraucher',
'ottonormalverbraucherin',
'staatsbürger',
'staatsbürgerin',
'staatsvolk',
'steuerpflichtige',
'steuerzahler',
'steuerzahlerin',
'verbraucher',
'verbraucherin',
'verbraucherinnen',
'volk',
'volksstimme',
'volksgenosse',
'volkskörper',
'wahlvolk',
'wähler',
'wählerin',
'zivilbevölkerung',
'zivilgesellschaft',
'weltgemeinschaft',
]

people_ger = [
    'bewohner',
    'einwohner',
    'gesellschaft',
    'gemeinschaft',
    'leute',
    'mann',
    'nation'
    'republik'
    'staatsbürger',
]

people_ordinary = [
    'leute',
    'mann',
    'mensch',
]


attr_ger = [
    'deutsch',
    'deutsche',
    'deutschlands',
    'einheimisch',
    'einheimische'
]

# %%
if __name__ == "__main__":
    import pandas as pd

    df_people = pd.DataFrame(people, columns=['feature'])
    df_people['type'] = 'people'

    df_people_ordinary = pd.DataFrame(people_ordinary, columns=['feature'])
    df_people_ordinary['type'] = 'people_ordinary'

    df_attr_ordinary = pd.DataFrame(attr_ordinary, columns=['feature'])
    df_attr_ordinary['type'] = 'attr_ordinary'

    df_people_ger = pd.DataFrame(people_ger, columns=['feature'])
    df_people_ger['type'] = 'people_ger'

    df_attr_ger = pd.DataFrame(attr_ger, columns=['feature'])
    df_attr_ger['type'] = 'attr_ger'

    df = df_people.append([df_people_ordinary, df_people_ger, df_attr_ordinary, df_attr_ger], ignore_index = True)

    df.to_csv('people_dict.csv', index=False)

# %%
