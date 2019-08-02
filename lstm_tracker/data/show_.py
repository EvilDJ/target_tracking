from pandas import read_csv
from matplotlib import pyplot as plt

dataset = read_csv('./ground.csv', header=0, index_col=0)
values = dataset.values

group = [0, 1, 2, 3]
i = 1
plt.figure()
for g in group:
    plt.subplot(len(group), 1, i)
    plt.plot(values[:, g])
    plt.title(dataset.columns[g], y=0.5, loc='right')
    i+=1
plt.show()