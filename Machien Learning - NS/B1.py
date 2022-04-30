import numpy as np
players = [180, 172, 178, 185, 190, 195, 192, 200, 210, 190]

a = np.array(players)

mean = np.sum(a)/a.size

v = np.sum((a-mean)**2)/a.size
count = 0
for i in players:
    if (i >= mean and i < v) or (i <= mean and i > v) :
        count += 1
print(count)