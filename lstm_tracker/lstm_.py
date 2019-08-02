from math import sqrt,ceil
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.callbacks import LearningRateScheduler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import backend as K

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def data_change(data, split):
    out_x = []
    _x = []
    out_y = []
    j = 1
    i = 1
    while len(data)>= i:
        _x.append(data[i -1])
        # print(i ,'1111')
        if j == split:
            out_x.append(_x[0:split-1])
            out_y.append(_x[split-1])
            j = 0
            i -= (split - 1)
            _x = []
        i += 1
        j += 1
    return np.array(out_x), np.array(out_y)

def print_ser(data):
    j = 0
    for i in data:
        print(j, i )
        j += 1

def mloss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def scheduler(epoch):
    if epoch % 50 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
    return K.get_value(model.optimizer.lr)

def bb(x, y):
    x1, y1, w1, h1 = x
    x2, y2, w2, h2 = y
    coli = abs(min(x1 + w1, x2 + w2) - max(x1, x2))
    rowi = abs(min(y1 + h1, y2 + h2) - max(y1, y2))
    overi = coli * rowi
    area1 = w1 * h1
    area2 = w2 * h2
    return overi / (area1 + area2 - overi)

def acc(x, y):
    m = 0
    for i in range(len(x)):
        print(x[i])
        m += bb(x[i],y[i])
    return m / len(y)

dataset = read_csv('./data/ground.csv', header=0, index_col=0)
# values = dataset.values #读取csv文件的数据
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
n_train_hours = int(len(dataset) * 0.7)
train = dataset[:n_train_hours, :]
test = dataset[n_train_hours:, :]
s = 2
train_X, train_y = data_change(train, s)
test_X, test_y = data_change(test ,s)
_x,_y = data_change(dataset, s)
# print_ser(_x)

reduce_lr = LearningRateScheduler(scheduler)
model = Sequential()
model.add(LSTM(25, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(train_X.shape[2]))
model.compile(loss= mloss, optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=12, validation_data=(test_X, test_y),
                    verbose=2, shuffle=False, callbacks=[reduce_lr])
model.save('./model/my_model.h5')

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
# x1 = [198,214,34,81]
# x2 = [197,214,34,81]
# x3 = [195,214,34,81]
# x = np.array([x1, x2, x3])
# x = scaler.fit_transform(x)
# x = np.array([x])
yhat = model.predict(_x)
yhat = scaler.inverse_transform(yhat)
test_y = scaler.inverse_transform(_y)
yhat = np.around(yhat)
# print_ser(yhat)
with open('./data/lstm.txt','a+') as f:
    for i in yhat:
        for ii in range(len(i)):
            if ii < 3 :
                print(i[ii])
                f.write(str(i[ii]) + ',')
            else:f.write(str(i[ii]))
        f.write('\n')

print('test accuracy:', np.mean(acc(yhat, test_y)))


pyplot.plot(test_y, label="Actual " )
pyplot.plot(yhat, label="Predicted " )
pyplot.legend()
pyplot.show()