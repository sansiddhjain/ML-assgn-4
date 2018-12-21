import numpy as np
import pandas as pd 
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

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
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.reshape((len(y_train), 1))
y_val = y_val.reshape((len(y_val), 1))
y_train_one_hot = keras.utils.to_categorical(y_train, num_classes=20)
y_val_one_hot = keras.utils.to_categorical(y_val, num_classes=20)   

X_test = np.load('test/test.npy')
mean = np.mean(X_test, axis=0)
X_test = np.subtract(X_test, mean)
X_test = normalize(X_test)

df = pd.DataFrame(columns=['hidden_layer', 'train_acc', 'val_acc'])
count = 0

#Hyper-parameter-tuning
for hidden_layer in [16, 32, 64, 96, 128, 256, 512, 1024, 2048, 4096]:
	model = Sequential()
	print('Starting training on '+str(hidden_layer))
	model.add(Dense(units=hidden_layer, activation='sigmoid', input_dim=784))
	model.add(Dense(units=20, activation='softmax'))
	adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

	model.fit(X_train, y_train_one_hot, epochs=100, batch_size=128, verbose=2, validation_data=(X_val, y_val_one_hot))
	loss_and_metrics = model.evaluate(X_train, y_train_one_hot)
	train_acc = loss_and_metrics[1]
	loss_and_metrics = model.evaluate(X_val, y_val_one_hot)
	val_acc = loss_and_metrics[1]
	print(str(loss_and_metrics))
	df.loc[count, :] = [hidden_layer, train_acc, val_acc]
	count += 1

	df_sub = pd.read_csv('sampleSubmission.csv')
	print('Calculating predictions on test set...')
	y_pred = model.predict(X_test)
	print(y_pred)
	y_pred = np.argmax(y_pred, axis=1)
	print(y_pred)
	y_pred = y_pred.reshape((len(y_pred), 1))
	df_sub['CATEGORY'] = y_pred
	df_sub['CATEGORY'] = pd.Series(list(map(lambda x : id2class[x], df_sub['CATEGORY'])))
	df_sub.to_csv('nn/submission_nn_'+str(hidden_layer)+'.csv', index=False)
	df.to_csv('nn/nn_hyp_tuning.csv', index=False)
