import numpy as np
import os
from scipy.stats import expon,norm
from sklearn import metrics
from SimulationParameters import *

# for key, value in vars(SimulationParameters).items():
#     if not key.startswith('__'):  # Ignore special/magic attributes
#         globals()[key] = value


def Run(seed=seed): 
    # locals().update(vars(SimulationParameters))

    if seed is None:
        seed = int.from_bytes(os.urandom(4), 'little') # System Entropy based random seed generation
    np.random.seed(seed)
    
    pos_t = np.empty((p,d,nt+1))            
    pos_t[:,:, 0] = xInitial  
    state_t = np.empty((p,nt+1), dtype = int)            
    break_mean = eps / (c_switch * param)  
    # switchtime = expon.rvs(scale=break_mean, size=p, random_state = seed)
    switchtime = expon.rvs(scale=break_mean, size=p)
    binding_time_fraction = np.empty((p, p))
    pairwiseDistances = np.empty((p,p))
    potentialConfinements = np.empty((p, d, nt)) 
    excludedVolumes = np.zeros((p, d, nt))
    stochasticCrosslinkings = np.empty((p, d, nt))
    # noiseArray = norm.rvs(loc = 0, scale = 1, size = (p, d, nt+1), random_state = seed) * np.sqrt(2*dt*eps/phi)
    noiseArray = norm.rvs(loc = 0, scale = 1, size = (p, d, nt+1)) * np.sqrt(2*dt*eps/phi)


    # if printBeadActions:
        # print(pos_t.shape)
        # print(state_t.shape)


    # Initialize close beads bound
    state_t[:, 0] = np.arange(p) # Initial state 
    for k in range(p):
        for l in range(k + 1, p):
            if state_t[k, 0] == k and state_t[l, 0] == l and np.linalg.norm(xInitial[k,:] - xInitial[l,:]) < 1e-4:
                state_t[k, 0] = l
                state_t[l, 0] = k
                # b = expon.rvs(scale=break_mean, random_state = seed)
                b = expon.rvs(scale=break_mean)

                switchtime[k] = b
                switchtime[l] = b


    # Main simulation loop
    for step in range(1,nt+1):
        # print(f'step {step}')

        x = pos_t[:,:, step-1]
        state = state_t[:, step-1]
        t = dt * (step-1)


        # Update Bead Positions
        for k in range(p):
            stochasticCrosslinkings[k, :, step-1] = k_v * (x[state[k], :] - x[k, :]) 
            potentialConfinements[k, :, step-1] = -1 * k_c * x[k,:] * (np.linalg.norm(x[k,:]) ** 2)
            for j in range(p):
                if j == k: continue
                excludedVolumes[k, :, step-1] += a_ev * (x[k,:] - x[j,:]) * np.exp(-1 * (np.linalg.norm(x[k,:] - x[j,:]) ** 2) / c_ev) 
            v = stochasticCrosslinkings[k, :, step-1] + potentialConfinements[k, :, step-1] + excludedVolumes[k, :, step-1]
            drift = dt * (v / phi)

            noise = noiseArray[k,:,step]

            x[k] += (drift+noise)          
        # End of bead positions loop


        # Break Bonds
        visitedBeads = set()
        for k in range(p):            
            # print(f'currently visiting bead {k}')

            if k in visitedBeads: # No bead can act twice in a single timestep
                # print(f'bead {k} was already visited')
                continue
            
            if state[k] != k: # Bead k is bound
                partner = state[k] # Bead k is bound to partner bead
                visitedBeads.add(k)
                visitedBeads.add(partner)

                if switchtime[k] < (t + dt) and state[partner] == k and switchtime[k] == switchtime[partner]: # Bond breaks)

                    state[partner] = partner
                    state[k] = k

                    binding_time_fraction[k, partner] = (switchtime[k] - t) / dt 
                    binding_time_fraction[partner, k] = (switchtime[k] - t) / dt

                    # b = expon.rvs(scale=break_mean, random_state = seed)
                    switchtime[k] = np.nan
                    switchtime[partner] = np.nan
                    # used to be switchtime[each] = t + b
                    # if printBeadActions:
                        # print(f'the bond between beads {k} and {partner} has broken')
                        # print(f'{state}')

                else: # bead stays bound
                    binding_time_fraction[k, state[k]] = 1
            else: # bead k is unbound      
                 pass
        # End of Bond Breaking Loop
        

        # Get Bond Formation Times
        pairwiseDistances = metrics.pairwise_distances(x,x, metric='euclidean')
        formTimes = []
        for i in range(p):
            for j in range(i+1,p):
                affinityRate = affinity(pairwiseDistances[k][j], param)
                # sampleFormTime = expon.rvs(scale = (eps/affinityRate), random_state = seed)
                sampleFormTime = expon.rvs(scale = (eps/affinityRate))

                formTimes.append((sampleFormTime , i , j)) #Keep track of the formation time for 2 beads i and j
        formTimes = sorted(formTimes, key=lambda x: x[0]) #Sort based on formation time

        # Form Bonds
        for formTime in formTimes:
            if formTime[0] > dt: #all remaining times will also be > dt, so no bonds will form, break the loop
                break

            bead1 = formTime[1]
            bead2 = formTime[2]
            
            if bead1 in visitedBeads:
                continue

            if state[bead1] != bead1:
                visitedBeads.add(bead1)
                continue

            if bead2 in visitedBeads:
                continue

            if state[bead2] != bead2:
                visitedBeads.add(bead2)
                continue

            state[bead1] = bead2
            state[bead2] = bead1
            # break_time = t + formTime[0] + expon.rvs(scale=break_mean, random_state = seed)
            break_time = t + formTime[0] + expon.rvs(scale=break_mean)

            switchtime[bead1] = break_time
            switchtime[bead2] = break_time
            visitedBeads.add(bead1)
            visitedBeads.add(bead2)

            # if printBeadActions:
            #     # print(f' beads {bead1} and {bead2} form a bond')
            #     # print(f'{state}')

        # Save position and state at this time step
        pos_t[:,:,step] = x  
        state_t[:,step] = state
        # print()

    return stochasticCrosslinkings, potentialConfinements, excludedVolumes, noiseArray, pos_t, state_t, seed,


def Simulate(numSims = numSims):

    stochasticCrosslinkings = np.empty(shape = (p,d,nt,numSims))
    potentialConfinements = np.empty(shape = (p,d,nt,numSims))
    excludedVolumes = np.empty(shape = (p,d,nt,numSims))
    noiseArray = np.empty(shape = (p,d,nt+1,numSims))
    positions = np.empty(shape = (p,d,nt+1,numSims))
    states = np.empty(shape = (p,nt+1,numSims))
    seeds = np.empty(shape = numSims)


    for i in range(numSims):
        currentSimulationRun = Run()

        stochasticCrosslinkings[:,:,:, i] = currentSimulationRun[0]
        potentialConfinements[:,:,:,i] = currentSimulationRun[1]
        excludedVolumes[:,:,:, i] = currentSimulationRun[2]
        noiseArray[:, :, :, i] = currentSimulationRun[3]
        positions[:,:,:,i] = currentSimulationRun[4]
        states[:,:, i] = currentSimulationRun[5]
        seeds[i] = currentSimulationRun[6]

    return stochasticCrosslinkings, potentialConfinements, excludedVolumes, noiseArray, positions, states, seeds