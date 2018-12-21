import pandas as pd

df1 = pd.read_csv('vgg13_1.csv')
df2 = pd.read_csv('vgg13_2.csv')
df3 = pd.read_csv('kurapan_mnist.csv')
df4 = pd.read_csv('kurapan2.csv')
df5 = pd.read_csv('vgg3.csv')
# df6 = pd.read_csv('vgg13_3.csv')
# df7 = pd.read_csv('vgg_kaggle.csv')

df_id = pd.read_csv('id2class.csv')
id2class = dict(zip(df_id['ID'], df_id['Class']))
class2id = dict(zip(df_id['Class'], df_id['ID']))

df1['CATEGORY'] = pd.Series(list(map(lambda x : class2id[x], df1['CATEGORY'])))
df2['CATEGORY'] = pd.Series(list(map(lambda x : class2id[x], df2['CATEGORY'])))
df3['CATEGORY'] = pd.Series(list(map(lambda x : class2id[x], df3['CATEGORY'])))
df4['CATEGORY'] = pd.Series(list(map(lambda x : class2id[x], df4['CATEGORY'])))
df5['CATEGORY'] = pd.Series(list(map(lambda x : class2id[x], df5['CATEGORY'])))
# df6['CATEGORY'] = pd.Series(list(map(lambda x : class2id[x], df6['CATEGORY'])))
# df7['CATEGORY'] = pd.Series(list(map(lambda x : class2id[x], df7['CATEGORY'])))

df_m = df1.merge(df2, on='ID', how='inner')
df_m = df_m.merge(df3, on='ID', how='inner')
df_m = df_m.merge(df4, on='ID', how='inner')
df_m = df_m.merge(df5, on='ID', how='inner')
# df_m = df_m.merge(df7, on='ID', how='inner')
# df_m = df_m.merge(df6, on='ID', how='inner')
del df_m['ID']
df_ens = df_m.mode(axis=1)
df_ens.reset_index(inplace=True)
# print df_ens
df_ens = df_ens.iloc[:, :2]
df_ens.columns = ['ID', 'CATEGORY']
df_ens['CATEGORY'] = pd.Series(list(map(int, df_ens['CATEGORY'])))
df_ens['CATEGORY'] = pd.Series(list(map(lambda x : id2class[x], df_ens['CATEGORY'])))
df_ens.to_csv('ensemble8.csv', index=False)