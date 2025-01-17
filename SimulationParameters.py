import numpy as np
from math import exp

nt = 10
dt = 0.01
p = 3
d = 2
xInitial = np.zeros((p, d)) 
# seed = 3846706489  # this is a large triangle
seed = None
numSims = 1
param = 1
eps = 0.01

affinity = lambda x, param: param * 2 / (1 + exp(20 * (abs(x) - 0.75)))
stopflags = {'planes': [ {'normal': np.array([[0.0], [-0.20843747], [0.0], [0.78961351], [0.0], [-0.57711726]]), 'bias': np.array([[-0.36824147]])}  ,  {'normal': np.array([[0.0], [0.78961351], [0.0], [-0.20843747], [0.0], [-0.57711726]]),'bias': np.array([[-0.36824147]])}]}
sep = 0.92
k_v = 5
k_c = 1
a_ev = 2
c_ev = 0.5
c_switch = 0.5
phi = 0.8



class Simulation():
    def __init__(self, *args):
        (
            self.stochasticCrosslinkings,
            self.potentialConfinements,
            self.excludedVolumes,
            self.noiseArray,
            self.positions,
            self.states,
            self.seeds
        ) = args

    def printSeeds(self):
        for i in range(self.numSims):
            print(f'Seed {i+1}: {int(self.seeds[i])}')


def includeParametersInSimulation():
    for key, value in globals().items():
        if not key.startswith('__') and not callable(value):  
            setattr(Simulation, key, value)

includeParametersInSimulation()