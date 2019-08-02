from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
import numpy as np

def mloss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

model = load_model('./model/my_model.h5',custom_objects={'mloss':mloss})
scaler = MinMaxScaler(feature_range=(0, 1))
x1 = [198,214,34,81]
x2 = [197,214,34,81]
x3 = [195,214,34,81]
x = np.array([x1])
x = scaler.fit_transform(x)
x = np.array([x])

y = model.predict(x)
y = scaler.inverse_transform(y)
print(np.round(y))