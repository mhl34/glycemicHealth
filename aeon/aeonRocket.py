import aeon
print(dir(aeon.utils))
from aeon.transformations.collection.rocket import Rocket
from aeon.datasets import load_unit_test
X_train, y_train = load_unit_test(split="train")
X_test, y_test = load_unit_test(split="test")
trf = Rocket(num_kernels=512)
trf.fit(X_train)

X_train = trf.transform(X_train)
X_test = trf.transform(X_test)