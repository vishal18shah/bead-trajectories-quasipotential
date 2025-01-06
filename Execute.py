from BeadModel import Simulate
from SimulationParameters import *
from MakePlots import plotPositions, plotDistances
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time # A single simulation run takes < half a second so 100 runs takes 50 seconds. Seeding takes 2/3 of the time in each run. 

sim = Simulation(*Simulate(numSims))

sim.printSeeds()
# print(sim.states.shape)
# print(sim.states[:, :50, 0].T)


plotPositions(positions = sim.positions, numPlots = 9) #assert 1 <= numPlots <= numSims
plotDistances(positions = sim.positions, numPlots = 9)

