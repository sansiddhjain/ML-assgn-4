import numpy as np 
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score

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

print('Starting PCA...')
pca = PCA(n_components=50)
pca.fit(X)
X_pca = pca.transform(X)
X_pca = normalize(X_pca)
print('Finished PCA!')

# print('Starting Training...')
# clf = SVC(C=1.0, kernel='linear', verbose=True)
# clf.fit(X_pca, y)
# print('Finished Training!')
# acc = clf.score(X_pca, y)
# print "Training Accuracy - "+str(acc)

X_test = np.load('test/test.npy')
X_test_pca = pca.transform(X_test)
X_test_pca = normalize(X_test_pca)

df = pd.DataFrame(columns=['C', 'cross_val_score'])
count = 0
df = pd.read_csv('svm_hyp_tuning.csv')
count = len(df)

# for C in [1e-2, 1, 5, 10, 20]:
for C in [100]:
	print('Starting training on '+str(C))
	clf = SVC(C=C, kernel='linear', verbose=True)
	score = cross_val_score(clf, X_pca, y, cv=5)
	cvs = np.mean(score)

	df.loc[count, :] = [C, cvs]
	count += 1
	df.to_csv('svm_hyp_tuning.csv', index=False)

	df_sub = pd.read_csv('sampleSubmission.csv')
	print('Calculating predictions on test set...')
	clf.fit(X_pca, y)
	df_sub['CATEGORY'] = clf.predict(X_test_pca)
	df_sub['CATEGORY'] = pd.Series(list(map(lambda x : id2class[x], df_sub['CATEGORY'])))
	df_sub.to_csv('submission_svm_'+str(count)+'.csv', inplace=True)
	