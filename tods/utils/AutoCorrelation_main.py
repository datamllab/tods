import numpy as np
from skinterface.primitiveSKI.AutoCorrelation_skinterface import AutoCorrelationSKI

X = np.array([[3., 4., 8., 16, 18, 13., 22., 36., 59., 128, 62, 67, 78, 100]])

transformer = AutoCorrelationSKI(use_columns=(0,))
results = transformer.produce(X)

print("Primitive: ", transformer.primitive)
print("Results: \n", results)

