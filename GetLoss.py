import numpy as np
import torch
from SimulationParameters import *

def getGradU(): # Position tensor and model may be function parameters
    # Code Here
    # torch.autograd or something like that?
    # Returns the gradient of the potential function with respect to each position
    # gradU = torch.empty(3, 2)
    # return gradU
    return


def getResidue(x, gradU):
    # Single Timestep
    #both data(positions) and gradU(gradient of potential wrt positions) should be 3x2 tensors. in the general case, they are p by d in shape
    v1 = torch.zeros(3, 2)
    for k in range(3):
        v1[k, :] += -1 * k_c * x[k,:] * (torch.norm(x[k,:]) ** 2)
        for j in range(3):
            if j == k: continue
            v1[k, :] += a_ev * (x[k,:] - x[j,:]) * torch.exp(-1 * (torch.norm(x[k,:] - x[j,:]) ** 2) / c_ev) 

    v2 = v1 +  torch.tensor([ [k_v * (x[1, :] - x[2, :])],
                              [k_v * (x[2, :] - x[1, :])],
                              [      0    ,         0   ]
                            ])

    v3 = v1 +  torch.tensor([ [k_v * (x[1, :] - x[3, :])],
                              [      0    ,         0   ],
                              [k_v * (x[3, :] - x[1, :])]
                            ])
    
    v4 = v1 +  torch.tensor([ [      0    ,         0   ],
                              [k_v * (x[2, :] - x[3, :])],
                              [k_v * (x[3, :] - x[2, :])]
                            ])
    
    advectionForces = [v1, v2, v3, v4]
    advectionMatrix = torch.zeros(4, 4)
    diffusionMatrix = torch.zeros(4, 4)

    for i in range(4):
        advectionMatrix[i,i] = torch.sum(advectionForces[i] * gradU) # * does element-wise multiplication
        diffusionMatrix[i,i] = torch.sum(gradU * gradU)


    # make sure dimensionality is correct for torch.linalg.norm
    switchingMatrix = torch.tensor([ [                       0                             ,c_switch,c_switch ,c_switch ],
                                     [   affinity((torch.linalg.norm(x[1,:] - x[2,:])), 1)   ,-c_switch,  0  ,  0  ],
                                     [   affinity((torch.linalg.norm(x[1,:] - x[3,:])), 1)   ,  0  ,-c_switch,  0  ],
                                     [   affinity((torch.linalg.norm(x[2,:] - x[3,:])), 1)   ,  0  ,  0  ,-c_switch],
                                   ])
    switchingMatrix *= param

    switchingMatrix00 = -1 * sum(switchingMatrix[0, :])
    switchingMatrix[0][0] = switchingMatrix00
    # CHECK THAT value00 was updated correctly

    
    M = advectionMatrix + diffusionMatrix + switchingMatrix
    residue = torch.det(M)
    return residue #see what it looks like written out. remember how to take det's of 4x4 matrices
   

def getLoss(positions): #One simulation at a time so positions has shape (p,d,nt+1).
    loss = torch.mean(np.array([getResidue(positions[:, :, t]) for t in range(positions.shape[2])]) ** 2 )