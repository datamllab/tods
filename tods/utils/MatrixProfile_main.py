import numpy as np
from skinterface.primitiveSKI.MatrixProfile_skinterface import MatrixProfileSKI

X_train = np.array([[3., 4., 8., 16, 18, 13., 22., 36., 59., 128, 62, 67, 78, 100]])
X_test = np.array([[3., 4., 8.6, 13.4, 22.5, 17, 19.2, 36.1, 127, -23, 59.2]])

transformer = MatrixProfileSKI()
transformer.fit(X_train)
prediction_labels = transformer.predict(X_test)
prediction_score = transformer.predict_score(X_test)

print("Primitive: ", transformer.primitive)
print("Prediction Labels\n", prediction_labels)
print("Prediction Score\n", prediction_score)
