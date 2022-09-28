import numpy as np
from tods.sk_interface.detection_algorithm.AKAutoEncoder_skinterface import AKAutoEncoderSKI
from tods.sk_interface.detection_algorithm.AKRNN_skinterface import AKRNNSKI
from tods.sk_interface.detection_algorithm.AutoEncoder_skinterface import AutoEncoderSKI
from tods.sk_interface.detection_algorithm.VariationalAutoEncoder_skinterface import VariationalAutoEncoderSKI
from tods.sk_interface.detection_algorithm.DeepLog_skinterface import DeepLogSKI
from tods.sk_interface.detection_algorithm.Telemanom_skinterface import TelemanomSKI
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn import metrics

#prepare the data
# data = np.loadtxt("./ucr/005_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature1_4000_5391_5392.txt")
data = np.loadtxt("./ucr/012_UCR_Anomaly_DISTORTEDECG2_15000_16000_16100.txt")
# data = np.loadtxt("./ucr/019_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z1_5000_6168_6212.txt")

# data = np.loadtxt("./omni/test/machine-1-1.txt", delimiter=',')
# label = np.loadtxt("./omni/label/machine-1-1.txt")

# data = np.loadtxt("./omni/test/machine-2-1.txt", delimiter=',')
# label = np.loadtxt("./omni/label/machine-2-1.txt")

# data = np.loadtxt("./omni/test/machine-3-1.txt", delimiter=',')
# label = np.loadtxt("./omni/label/machine-3-1.txt")

# data = data[:28000]
# label = label[:28000]
# data_copy = data
# X_train = data


print(data.shape)
# X_train = data[:10000]
# X_test = data[10000:]

X_train = np.expand_dims(data[:15000], axis=1)
X_test = np.expand_dims(data[15000:30000], axis=1)

# X_train = np.expand_dims(   #this is for convblock and rnn
#     X_train, axis=2
# )
# labels = np.expand_dims(data[5000:10000], axis=1)
y_test = np.zeros(15000,)
y_test[1000:1100] = 1
print(y_test)

# transformer = AutoEncoderSKI()
transformer = AKAutoEncoderSKI()
transformer.fit(X_train)

prediction_labels_train = transformer.predict(X_train)

prediction_labels = transformer.predict(X_test)
# prediction_score = transformer.predict_score(X_test)

print("Prediction Labels\n", prediction_labels)
# print("Prediction Score\n", prediction_score)

# y_true = prediction_labels_train[:1000]
# y_pred = prediction_labels[:1000]

y_true = y_test
y_pred = prediction_labels

# y_pred = prediction_labels_train
# y_true = label

print(y_true.shape, y_pred.shape)

# real_shape = y_true.shape[0]

# y_pred = y_pred[:real_shape]

print('Accuracy Score: ', accuracy_score(y_true, y_pred))

confusion_matrix(y_true, y_pred)

print(classification_report(y_true, y_pred))

precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
f1_scores = 2*recall*precision/(recall+precision)

print('Best threshold: ', thresholds[np.argmax(f1_scores)])
print('Best F1-Score: ', np.max(f1_scores))

fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
roc_auc = metrics.auc(fpr, tpr)

print('AUC_score: \n', roc_auc)

plt.title('ROC')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#---------------------------------------------------------

#prepare the data
# data = np.loadtxt("./ucr/005_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature1_4000_5391_5392.txt")
# data = np.loadtxt("./ucr/012_UCR_Anomaly_DISTORTEDECG2_15000_16000_16100.txt")
# data = np.loadtxt("./ucr/019_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z1_5000_6168_6212.txt")

# data = np.loadtxt("./omni/test/machine-1-1.txt", delimiter=',')
# label = np.loadtxt("./omni/label/machine-1-1.txt")

# data = np.loadtxt("./omni/test/machine-2-1.txt", delimiter=',')
# label = np.loadtxt("./omni/label/machine-2-1.txt")

# data = np.loadtxt("./omni/test/machine-3-1.txt", delimiter=',')
# label = np.loadtxt("./omni/label/machine-3-1.txt")

# data = np.loadtxt("./machine-1-2-test.txt", delimiter=',')
# label = np.loadtxt("./machine-1-2-label.txt")

# data = data[:28000]
# label = label[:28000]
data_copy = data
X_train = data


print(data.shape)
# X_train = data[:10000]
# X_test = data[10000:]

# X_train = np.expand_dims(data[:15000], axis=1)
# X_test = np.expand_dims(data[15000:30000], axis=1)

# X_train = np.expand_dims(   #this is for convblock and rnn
#     X_train, axis=2
# )
# labels = np.expand_dims(data[5000:10000], axis=1)
# y_test = np.zeros(15000,)
# y_test[1000:1100] = 1
# print(y_test)

# transformer = AutoEncoderSKI()
transformer = AKAutoEncoderSKI()
transformer.fit(X_train)

prediction_labels_train = transformer.predict(X_train)

# prediction_labels = transformer.predict(X_test)
# prediction_score = transformer.predict_score(X_test)

# print("Prediction Labels\n", prediction_labels)
# print("Prediction Score\n", prediction_score)

# y_true = prediction_labels_train[:1000]
# y_pred = prediction_labels[:1000] 

y_true = label
y_pred = prediction_labels_train

# y_pred = prediction_labels_train
# y_true = label

print(y_true.shape, y_pred.shape)

# real_shape = y_true.shape[0]

# y_pred = y_pred[:real_shape]

print('Accuracy Score: ', accuracy_score(y_true, y_pred))

confusion_matrix(y_true, y_pred)

print(classification_report(y_true, y_pred))

precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
f1_scores = 2*recall*precision/(recall+precision)

print('Best threshold: ', thresholds[np.argmax(f1_scores)])
print('Best F1-Score: ', np.max(f1_scores))

fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
roc_auc = metrics.auc(fpr, tpr)

print('AUC_score: \n', roc_auc)

plt.title('ROC')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()