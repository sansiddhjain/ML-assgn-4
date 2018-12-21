import numpy as np
import pandas as pd 
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
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
# one_hot_labels = keras.utils.to_categorical(y, num_classes=20)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
y_train = y_train.reshape((len(y_train), 1))
X_val = X_val.reshape((X_val.shape[0], 28, 28, 1))
y_val = y_val.reshape((len(y_val), 1))
y_train_one_hot = keras.utils.to_categorical(y_train, num_classes=20)
y_val_one_hot = keras.utils.to_categorical(y_val, num_classes=20)	

X_test = np.load('test/test.npy')
mean = np.mean(X_test, axis=0)
X_test = np.subtract(X_test, mean)
X_test = normalize(X_test)
X_test = X_test.reshape((100000, 28, 28, 1))

df = pd.DataFrame(columns=['filter', 'kernel', 'train_acc', 'val_acc'])
count = 0
for filters in [64]:
	for kernel in [4]:
		print('Starting training on '+str(filters)+', '+str(kernel))
		model = Sequential()
		model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(28, 28, 1), padding='same'))
		# model.add(ZeroPadding2D(padding=(2, 2)))
		model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		# model.add(ZeroPadding2D(padding=(2, 2)))
		model.add(Conv2D(256, (5, 5), padding='same'))
		# model.add(ZeroPadding2D(padding=(1, 1)))
		model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.2))
		# model.add(ZeroPadding2D(padding=(1, 1)))
		model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
		model.add(Dropout(0.2))
		# model.add(ZeroPadding2D(padding=(1, 1)))
		model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.5))
		model.add(Flatten())
		model.add(Dense(units=2048, activation='relu'))
		model.add(Dense(20, activation='softmax'))
		adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
		model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

		model.fit(X_train, y_train_one_hot, epochs=30, batch_size=128, verbose=2, validation_data=(X_val, y_val_one_hot))
		loss_and_metrics = model.evaluate(X_train, y_train_one_hot)
		print(str(loss_and_metrics))
		train_acc = loss_and_metrics[1]
		loss_and_metrics = model.evaluate(X_val, y_val_one_hot)
		val_acc = loss_and_metrics[1]
		print(str(loss_and_metrics))
		df.loc[count, :] = [filters, kernel, train_acc, val_acc]
		count += 1
		df_sub = pd.read_csv('sampleSubmission.csv')
		print('Calculating predictions on test set...')
		y_pred = model.predict(X_test)
		# print(y_pred)
		y_pred = np.argmax(y_pred, axis=1)
		# print(y_pred)
		y_pred = y_pred.reshape((len(y_pred), 1))
		df_sub['CATEGORY'] = y_pred
		df_sub['CATEGORY'] = pd.Series(list(map(lambda x : id2class[x], df_sub['CATEGORY'])))
		df_sub.to_csv('comp/submission_cnn_mnist.csv', index=False)
		# df.to_csv('cnn_hyp_tuning.csv', index=False)