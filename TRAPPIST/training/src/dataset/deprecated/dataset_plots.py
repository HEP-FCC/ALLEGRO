import matplotlib.pyplot as plt
import networkx as nx
import plotly
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch
#interactive plots
fig = make_subplots(rows=2, cols=1, specs=[[{'type': 'scatter3d'}], [{'type': 'scatter3d'}]])
fig.add_trace(go.Scatter3d(x=pos[:,0], y=pos[:,1], z=pos[:,2], mode='markers',
    marker=dict(size=e_particle, color=particle_id, opacity=0.6, symbol = 'cross' )), row=1, col=1)
fig.add_trace(go.Scatter3d(x=pos_particles[:, 0], y=pos_particles[:, 1], z=pos_particles[:, 2],
                            mode='markers', marker = dict (size = e_particle, color = particle_id,opacity= 0.5,symbol = "x")), row=2, col=1)

scatter1 = go.Scatter3d(x=pos[:, 0], y=pos[:, 1],z=pos[:, 2], mode='markers', marker=dict(size=e_particle, color=particle_id, opacity=0.6, symbol = 'cross' )
)
scatter2 = go.Scatter3d(x=pos_particles[:, 0],y=pos_particles[:, 1],z=pos_particles[:, 2],mode='markers',
    marker=dict(size=particle_id, color=particle_id, opacity=0.5 , symbol = 'diamond')
)
data = [scatter1, scatter2]
layout = go.Layout(scene=dict(aspectmode='data'))
fig1 = go.Figure(data=data, layout=layout)

pyo.plot(fig, filename='plot1.html')
pyo.plot(fig1, filename = 'plot2.html')
#arrow plots
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
# Plot 3D arrows
z = pos_particles[:, 0]
x = pos_particles[:, 1]
c = pos_particles[:, 2]
particle_id_1 = particle_id - 1
# Use pos_particles as the starting points for the arrows
start_x = pos_particles[:, 0]
start_y = pos_particles[:, 1]
start_z = pos_particles[:, 2]
# Direction of arrows pointing to pos
arrow_x = pos[:, 0] - pos_particles[:, 0]
arrow_y = pos[:, 1] - pos_particles[:, 1]
arrow_z = pos[:, 2] - pos_particles[:, 2]
ax.quiver(start_x, start_y, start_z, arrow_x, arrow_y, arrow_z, color='r', alpha=0.1)
# Plot 3D scatter
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], marker='o', c=particle_id, alpha=0.1)
ax.scatter(pos_particles[:, 0], pos_particles[:, 1], pos_particles[:, 2], marker='*', s=50, alpha=0.5)
ax.set_xlim([-0.5, 0.3])
ax.set_ylim([-1,1])
ax.set_zlim([-0.5, 0.5])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

