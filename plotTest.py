import numpy as np
import matplotlib.pyplot as plt

# generate random data for plotting
x = np.linspace(0.0,100,20)

# now there's 3 sets of points
y1 = np.random.normal(scale=0.2,size=20)
y2 = np.random.normal(scale=0.5,size=20)
y3 = np.random.normal(scale=0.8,size=20)

# plot the 3 sets
plt.plot(x,y1,label='plot 1')
plt.plot(x,y2, label='plot 2')
plt.plot(x,y3, label='plot 3')

# call with no parameters
plt.legend()

plt.show()