import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1/(1 + np.exp(-x))

p_x = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [7, 6], [7, 7], [7, 8], [7, 9], [7, 10], [8, 6], [8, 7], [8, 8], [8, 9], [8, 10], [9, 6], [9, 7], [9, 8], [9, 9], [9, 10], [10, 6], [10, 7], [10, 8], [10, 9], [10, 10], [11, 6], [11, 7], [11, 8], [11, 9], [11, 10], [12, 6], [12, 7], [12, 8], [12, 9], [12, 10]])
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
plt.figure()
for i in range(len(p_x)):
    if y[i] == 1:
        plt.plot(p_x[i][0], p_x[i][1], 'bo')
    else:
        plt.plot(p_x[i][0], p_x[i][1], 'ro')

w = np.array([1, 0])
b = 0
a = np.zeros(60)
f = np.zeros(60)
p = np.zeros(60)

for j in range(300):
  for i in range(len(p_x)):
    a[i] = np.dot(w, p_x[i]) + b
    f[i] = sigmoid(a[i])
    p[i] = f[i]
    w = w + (y[i] - p[i])*p_x[i]
    b = b + (y[i] - p[i])

line_x = [0, 12]
line_y = [0, 0]

for i in range(len(line_x)):
  line_y[i] = (-w[0] * line_x[i] - b)/w[1]
 
plt.plot(line_x, line_y)
plt.savefig("picture.png")
