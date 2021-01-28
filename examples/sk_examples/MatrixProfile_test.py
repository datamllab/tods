import numpy as np
from sklearn.model_selection import train_test_split
from tods.tods_skinterface.primitiveSKI.detection_algorithm.MatrixProfile_skinterface import MatrixProfileSKI

#prepare the data

data = np.loadtxt("./500_UCR_Anomaly_robotDOG1_10000_19280_19360.txt")

print("shape:", data.shape)
print("First 5 rows:\n", data[:5])

X_train, X_test = train_test_split(data, test_size=0.2, random_state=0)

print("First 5 rows train:\n", X_train[:5])
print("First 5 rows test:\n", X_test[:5])

# X_train = np.array([[3., 4., 8., 16, 18, 13., 22., 36., 59., 128, 62, 67, 78, 100]])
# X_test = np.array([[3., 4., 8.6, 13.4, 22.5, 17, 19.2, 36.1, 127, -23, 59.2]])

# transformer = MatrixProfileSKI()
# transformer.fit(X_train)
# prediction_labels = transformer.predict(X_test)
# prediction_score = transformer.predict_score(X_test)

# print("Primitive: ", transformer.primitive)
# print("Prediction Labels\n", prediction_labels)
# print("Prediction Score\n", prediction_score)
