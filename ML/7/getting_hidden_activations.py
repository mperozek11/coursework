import numpy as np
from keras.models import Sequential
from keras.layers import Dense

number_input_nodes = 2
number_hidden_nodes = 3
number_output_nodes = 1

def loadData():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    return X, y

# Build a model with 2 input nodes, 3 hidden nodes, and 1 output node
def buildModel(X, y):
    model = Sequential()
    model.add(Dense(number_hidden_nodes, input_dim=number_input_nodes, activation='sigmoid'))
    model.add(Dense(number_output_nodes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Build a new model that won't be trained.
# This new model just loads the trained weights from the full model
# This full model is passed into this function as input
# Then when we are adding a Dense layer, we add the weights argument
# with the full model's learned weights from the first layer.
# Notice the model stops after one layer.  When we use this smaller model
# to make predictions, the outputs will be the original model's hidden layer
# activations.
def buildHalfModel(model):
    model2 = Sequential()
    model2.add(Dense(number_hidden_nodes, input_dim=number_input_nodes,
                weights=model.layers[0].get_weights(), activation='sigmoid'))
    model2.compile(loss='binary_crossentropy', optimizer='adam')
    return model2


X, y = loadData()
model = buildModel(X, y)
model.fit(X, y, epochs=100, batch_size=10)
_, accuracy = model.evaluate(X, y)
print(accuracy)

# Using the full trained model, create the half model
# that is not trained.  It just loads some of the weights
# from the original model and then makes predictions.
# The output from the predict function contains all the hidden
# layer activations from the trained model.
model2 = buildHalfModel(model)
activations = model2.predict(X)

# Dump out the activations.
# Notice there are 3 activations per input pattern.
# These are the 3 activations for the 3 hidden nodes in the network.
# If you change the number of hidden nodes, then the number of activations
# you get as output will change.
print(activations)





