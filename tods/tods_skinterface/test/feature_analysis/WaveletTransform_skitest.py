import numpy as np
from tods.tods_skinterface.primitiveSKI.feature_analysis.WaveletTransform_skinterface import WaveletTransformSKI

X_test = np.array([[3., 4., 8.6, 13.4, 22.5, 17, 19.2, 36.1, 127, -23, 59.2]])

transformer = WaveletTransformSKI()
X_transform = transformer.produce(X_test)

print("Primitive:", transformer.primitive)
print("X_transform:\n", X_transform)
