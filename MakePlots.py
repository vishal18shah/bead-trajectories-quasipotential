
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from itertools import combinations
from BeadModel import Simulate
from SimulationParameters import p,nt

def plotBeadPositions(positions, simIndex, ax, title):
    x = positions[:,0,:,simIndex]
    y = positions[:,1,:,simIndex]

    # Create the figure and axis
    # fig, ax = plt.subplots()
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.set_xticks([-1,-0.5,0,0.5,1])
    ax.set_yticks([-1,-0.5,0,0.5,1])
    ax.set_title("Bead Positions")

    # Initialize bead positions
    beads = [ax.plot([], [], 'o', markersize=8, label=f"Bead {i+1}")[0] for i in range(p)]
    trajectories = [ax.plot([], [], '-', lw=1, alpha=0.7)[0] for i in range(p)]
    ax.legend()

    # Initialize function for animation
    def init():
        for bead, trajectory in zip(beads, trajectories):
            bead.set_data([], [])
            trajectory.set_data([], [])
        return beads + trajectories

    # Update function for animation
    def update(frame):
        for i in range(p):
            # Update particle position
            beads[i].set_data(x[i][frame], y[i][frame])
            # Update trajectory
            trajectories[i].set_data(x[i][:frame+1], y[i][:frame+1])
        return beads + trajectories

    # Create animation
    ani = animation.FuncAnimation(ax.figure, update, frames=len(x[0]), init_func=init, blit=True, interval=75) #ax.figure used to be fig
    # plt.xticks([-1,-0.5,0,0.5,1])
    # plt.yticks([-1,-0.5,0,0.5,1])
    # return fig,ani
    return ani


def plotPositions(positions, numPlots):
    # figureAnimations = [plotBeadPositions(positions,simIndex) for simIndex in range(numPlots)]
    # plt.show()
    multiplePlots = numPlots > 1

    rows = cols = int(numPlots ** (1/2)) 
    fig, axes = plt.subplots(rows, cols, figsize=(4*rows, 4*cols))
    fig.suptitle("Bead Position Animations", fontsize=16, weight='bold')


    if multiplePlots: axes = axes.flatten()
    animations = []
    for simIndex in range(numPlots):  # Handle cases where numPlots < 4
        if multiplePlots:
            ani = plotBeadPositions(positions, simIndex, axes[simIndex], title = f"Animation {simIndex + 1}")
        else:
            ani = plotBeadPositions(positions, simIndex, axes, title = f"Animation {simIndex + 1}")
        animations.append(ani)
    plt.tight_layout()
    plt.show()


def plotBeadDistances(positions, simIndex):
    xy = positions[:,:,:,simIndex]
    pairs = list(combinations(range(p), 2))  # All unique pairs
    distances = np.zeros(shape= (nt+1, len(pairs)))
    for time in range(nt+1):
        distances[time] = np.array([np.linalg.norm(xy[pair[0],:,time] - xy[pair[1],:,time]) for pair in pairs])  
    return distances


def plotDistances(positions, numPlots):
    multiplePlots = numPlots > 1

    rows = cols = int(numPlots ** (1/2)) 
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if multiplePlots: axes = axes.flatten()  # Flatten the 2D array of axes for easy indexing
    fig.suptitle("Bead Distances", fontsize=16, weight='bold')


    for simIndex in range(numPlots):
        distances = plotBeadDistances(positions, simIndex)
        ax = axes[simIndex] if multiplePlots else axes
        for pairIndex in range(distances.shape[1]):
            ax.plot(range(nt + 1), distances[:, pairIndex], label=f'Pair {pairIndex + 1}')
        ax.set_title(f"Simulation {simIndex + 1}")
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Distance')
        ax.set_xlim(0, nt)  
        ax.set_ylim(0, 2.5)
        ax.set_yticks([0, 0.5, 1, 1.5, 2])
        ax.legend()
    plt.tight_layout()
    plt.show()



