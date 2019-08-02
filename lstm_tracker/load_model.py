from keras.models import load_model
from math import sqrt,ceil
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


model = load_model('./model/my_model.h5')

def mloss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


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

def print_ser(data):
    j = 0
    for i in data:
        print(j, i )
        j += 1

dataset = read_csv('./data/ground.csv', header=0, index_col=0)
values = dataset.values #读取csv文件的数据
reframed = series_to_supervised(values, 1, 1)
values = reframed.values
n_train_hours = ceil(560)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
train_X, train_y = train[:, :4], train[:, 4:]
test_X, test_y = test[:, :4], test[:, 4:]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

yhat = model.predict(test_X, batch_size=32)
# print_ser(yhat)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
inv_yhat = concatenate((yhat, test_X[:, 4:]), axis=1)
inv_yhat = inv_yhat[:, :4]
test_y = test_y.reshape((len(test_y), 4))
inv_y = concatenate((test_y, test_X[:, 4:]), axis=1)
print_ser(inv_yhat)
inv_y = inv_y[:, :4]
# print_ser(inv_y)

pyplot.plot(inv_y, label="Actual " )
pyplot.plot(inv_yhat, label="Predicted " )
pyplot.legend()
pyplot.show()