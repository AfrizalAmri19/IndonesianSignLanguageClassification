from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
import os
import numpy as np

DATA_PATH = os.path.join('MP_Data')

actions = np.array(['hello','selamat','pagi','siang','sore','malam','sampai jumpa'])

no_sequences = 50

sequence_length = 60

label_map = {label:num for num, label in enumerate(actions)}
print(label_map)

sequences, labels = [],[]
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence),"{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences).shape
x = np.zeros(X)
norm = np.linalg.norm(x)
normal_array = x/norm
print(normal_array)
print(X)
y = to_categorical(labels).astype(int)
print(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.03)
print(y_test.shape)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

#CNN Conv-1D Deeplearning
verbose, epoch, batch_size = 2, 1000, 150

model = Sequential()
model.add(Conv1D(300, kernel_size=1, activation='linear', input_shape=(60, 258)))
model.add(Conv1D(128,  kernel_size=1, activation='linear'))
model.add(Conv1D(64,  kernel_size=1, activation='linear'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(32, activation='linear'))
model.add(Dense(18, activation='linear'))
model.add(Dense(actions.shape[0], activation='sigmoid'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs= epoch,verbose=verbose,  batch_size=batch_size, callbacks=[tb_callback])

print(model.summary())

#Predictions
res = model.predict(X_test)
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1). tolist()
yhat = np.argmax(yhat, axis=1).tolist()
m = multilabel_confusion_matrix(ytrue,yhat)
n = accuracy_score(ytrue,yhat)

model.save('IndonesiaSignLanguage.h5')
del model

#print(model.summary())
# print(m)
print(('Accuracy Score:'), n)
print(actions[np.argmax(res[4])])
print(actions[np.argmax(y_test[4])])
