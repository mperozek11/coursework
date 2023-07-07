import numpy as np
from keras.models import Sequential
from keras.layers import Dense

def loadData():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    return X, y

def buildModel(X, y):
    model = Sequential()
    model.add(Dense(10, input_dim=2, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


X, y = loadData()
model = buildModel(X, y)
model.fit(X, y, epochs=1500, batch_size=10)
_, accuracy = model.evaluate(X, y)
print(accuracy)





