from __future__ import division
import numpy as np 
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

print("Reading Data...")
f = os.listdir('train/')
f = [j for j in f if '.npy' in j]

cats = [j.split('.')[0] for j in f]
class2id = dict(list(zip(cats, np.arange(len(cats)))))
id2class = dict(list(zip(np.arange(len(cats)), cats)))

X = np.zeros((100000, 784))
y = np.zeros(100000)

for i in range(len(f)):
    file = f[i]
    a = np.load('train/'+file)
    X[5000*i:5000*(i+1), :] = a
    y[5000*i:5000*(i+1)] = class2id[cats[i]]
print("Read and formatted data!")

#Pre-processing
mean = np.mean(X, axis=0)
X = np.subtract(X, mean)
X = normalize(X)

X_test = np.load('test/test.npy')
mean = np.mean(X_test, axis=0)
X_test = np.subtract(X_test, mean)
X_test = normalize(X_test)

df_lst = pd.DataFrame(columns=['Max_Iter', 'Train_Acc'])
count = 0

for max_iter in [10, 20, 30, 40, 50]:
    print "Beginning Training with max_iter = "+str(max_iter)+"..."
    clf = KMeans(n_clusters=20, n_init=10, max_iter=max_iter, verbose=False)
    clf.fit(X)
    clusts = clf.predict(X)
    print(np.unique(clusts))
    print(y)

    df = pd.DataFrame(columns=['Cluster_id', 'True_label'])
    df['Cluster_id'] = clusts
    df['True_label'] = y

    grp = df.groupby('Cluster_id').agg(lambda x:x.value_counts().index[0])
    grp.reset_index(inplace=True)

    dic = dict(zip(grp['Cluster_id'], grp['True_label']))

    df['Pred_label'] = pd.Series(list(map(lambda x : dic[x], df['Cluster_id'])))

    acc = sum(df['True_label'] == df['Pred_label'])/(len(df))
    print acc

    df_lst.loc[count, :] = [max_iter, acc]
    count += 1

    print('Calculating predictions on test set...')

    mat = np.matmul(X_test, clf.cluster_centers_.T)
    clusters = np.argmax(mat, axis=1)
    y_pred = np.asarray(list(map(lambda x : dic[x], clusters)))
    y_pred = y_pred.reshape((len(y_pred), 1))
    df_sub = pd.read_csv('sampleSubmission.csv')
    df_sub['CATEGORY'] = y_pred
    df_sub['CATEGORY'] = pd.Series(list(map(lambda x : id2class[x], df_sub['CATEGORY'])))
    df_sub.to_csv('kmeans/submission_kmeans_'+str(max_iter)+'.csv', index=False)
    df_lst.to_csv('kmeans/kmeans_hyp_tuning.csv', index=False)
    print(str(max_iter)+' done.')
  
print "Beginning Training..."
clf = KMeans(n_clusters=20, n_init=10, max_iter=300, verbose=True)
clf.fit(X)
clusts = clf.predict(X)
# print(np.unique(clusts))
# print(y)

df = pd.DataFrame(columns=['Cluster_id', 'True_label'])
df['Cluster_id'] = clusts
df['True_label'] = y

grp = df.groupby('Cluster_id').agg(lambda x:x.value_counts().index[0])
# print grp
grp.reset_index(inplace=True)

dic = dict(zip(grp['Cluster_id'], grp['True_label']))
# print dic

df['Pred_label'] = pd.Series(list(map(lambda x : dic[x], df['Cluster_id'])))

# print (df['True_label'] == df['Pred_label'])
# print sum(df['True_label'] == df['Pred_label'])
# print len(df)
acc = sum(df['True_label'] == df['Pred_label'])/(len(df))
print acc

print('Calculating predictions on test set...')

mat = np.matmul(X_test, clf.cluster_centers_.T)
clusters = np.argmax(mat, axis=1)
y_pred = np.asarray(list(map(lambda x : dic[x], clusters)))
y_pred = y_pred.reshape((len(y_pred), 1))
df_sub = pd.read_csv('sampleSubmission.csv')
df_sub['CATEGORY'] = y_pred
df_sub['CATEGORY'] = pd.Series(list(map(lambda x : id2class[x], df_sub['CATEGORY'])))
df_sub.to_csv('kmeans/submission_kmeans.csv', index=False)
# print 'YAAAAY'