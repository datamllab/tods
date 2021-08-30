from autokeras.engine.block import Block
import autokeras as ak
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.util import nest
from numpy import asarray
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# from sklearn.model_selection import train_test_split
from autokeras import StructuredDataClassifier
from autokeras import StructuredDataRegressor
from tods.sk_interface.detection_algorithm.AutoEncoder_skinterface import AutoEncoderSKI

#how to split yahoo?
#is y true and y pred correct?
#show the notebook error
#how to get autokeras reports

#dataset
dataset = pd.read_csv("./yahoo_sub_5.csv")
data = dataset.to_numpy()
labels = dataset.iloc[:,6]
value1 = dataset.iloc[:,2] # delete later
print(labels)

#tods primitive
transformer = AutoEncoderSKI()
transformer.fit(data)
tods_output = transformer.predict(data)
prediction_score = transformer.predict_score(data)
print('result from AE primitive: \n', tods_output) #sk report
print('score from AE: \n', prediction_score)

#sk report
y_true = labels
y_pred = tods_output

print('Accuracy Score: ', accuracy_score(y_true, y_pred))

print('confusion matrix: \n', confusion_matrix(y_true, y_pred))

print(classification_report(y_true, y_pred))

precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
f1_scores = 2*recall*precision/(recall+precision)

print('Best threshold: ', thresholds[np.argmax(f1_scores)])
print('Best F1-Score: ', np.max(f1_scores))


#classifier
print('Classifier Starts here:')
search = StructuredDataClassifier(max_trials=15)
# perform the search
search.fit(x=data, y=labels, verbose=0) # y = data label colume
# evaluate the model
loss, acc = search.evaluate(data, labels, verbose=0)
print('Accuracy: %.3f' % acc)
# use the model to make a prediction
# row = [0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032]
# X_new = asarray([row]).astype('float32')
yhat = search.predict(data)
print('Predicted: %.3f' % yhat[0])
# get the best performing model
model = search.export_model()
# summarize the loaded model
model.summary()

#regressor
print('Regressor Starts here:')
search = StructuredDataRegressor(max_trials=15, loss='mean_absolute_error')
# perform the search
search.fit(x=data, y=labels, verbose=0) # y = data label
mae, _ = search.evaluate(data, labels, verbose=0)
print('MAE: %.3f' % mae)
# use the model to make a prediction
# X_new = asarray([[108]]).astype('float32')
yhat = search.predict(data)
print('Predicted: %.3f' % yhat[0])
# get the best performing model
model = search.export_model()
# summarize the loaded model
model.summary()
