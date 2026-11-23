import pandas as pd
import csv
import matplotlib.pyplot as plt


#cols = ['t', 'q1', 'q2']
df = pd.read_csv('snake.csv', header=None)

#t = df['t']
#y = df['y']

#print('Dimension de t', t.shape)
#print('Dimension de y', y.shape)

print(df.shape[1])

plt.figure(figsize=(12,5))

# 2. Create the plot using column names
plt.plot(df[1], df[2], '-b', label='Testing')

# 3. Add labels and a title
# plt.title('Angle variations')
# plt.xlabel('Time')
# plt.ylabel('Rads ($)')



fig1, ax1 = plt.subplots()
for i  in range(12, 23):
    #phi_required_plot = alphaA * np.sin(omega * sol.t + (i-1) * delta) + offset
    # ax1.plot(sol.t, phi_required_plot, label='($q_{i}$)')
    ax1.plot(df[0], df[i], label=f"q_{i}")
ax1.set_title('Reference Serpenoide Plot')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Angular Displacement(m)-Speed(rad)')
ax1.legend()
ax1.grid(True)



# plt.figure(figsize=(8, 4))
# plt.plot(df[1], df[2],label='Head Trajectory (x, y)')
# plt.xlabel('X Position (m)')
# plt.ylabel('Y Position (m)')
# plt.title(f'{9}-Link Snake Robot Locomotion')
# plt.axis('equal')
# plt.grid(True)
# plt.legend()
# plt.show()




# 4. Display the chart
plt.show()