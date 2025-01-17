import torch
import pandas as pd
from SimulationParameters import *

paths = torch.tensor(pd.read_csv('alpha1_path.txt', header = None).values)
gradients = torch.tensor(pd.read_csv('alpha1_dW.txt', header = None).values)

paths = paths.T.reshape(-1,3,2)
gradients = gradients.T.reshape(-1,3,2)

print(f"Phi: {phi}")

def getGradU(): # Position tensor and model may be function parameters
    # Code Here
    # torch.autograd or something like that?
    # Returns the gradient of the potential function with respect to each position
    # gradU = torch.empty(3, 2)
    # return gradU
    return


def getResidue(x, gradU):
    # Single Timestep
    #both x(positions) and gradU(gradient of potential wrt positions) should be 3x2 tensors. in the general case, they are p by d in shape
    v1 = torch.zeros(3, 2)
    for k in range(3):
        v1[k, :] += -1 * k_c * x[k,:] * (torch.norm(x[k,:]) ** 2)
        for j in range(3):
            if j == k: continue
            v1[k, :] += a_ev * (x[k,:] - x[j,:]) * torch.exp(-1 * (torch.norm(x[k,:] - x[j,:]) ** 2) / c_ev) 

    v2 = v1 + torch.stack([
    k_v * (x[1, :] - x[0, :]),
    k_v * (x[0, :] - x[1, :]),
    torch.tensor([0.0, 0.0])
])

    v3 = v1 + torch.stack([
        k_v * (x[2, :] - x[0, :]),
        torch.tensor([0.0, 0.0]),
        k_v * (x[0, :] - x[2, :])
    ])

    v4 = v1 + torch.stack([
        torch.tensor([0.0, 0.0]),
        k_v * (x[2, :] - x[1, :]),
        k_v * (x[1, :] - x[2, :])
    ])

    
    advectionForces = [v1, v2, v3, v4]
    advectionForces = [x / phi for x in advectionForces]
    advectionMatrix = torch.zeros(4, 4)
    diffusionMatrix = torch.zeros(4, 4)

    for i in range(4):
        advectionMatrix[i,i] = torch.sum(advectionForces[i] * gradU) # * does element-wise multiplication
        diffusionMatrix[i,i] = torch.sum(gradU * gradU) 

    diffusionMatrix /= phi

    # make sure dimensionality is correct for torch.linalg.norm
    switchingMatrix = torch.tensor([ [                       0                             ,c_switch,c_switch ,c_switch ],
                                     [   affinity((torch.linalg.norm(x[0,:] - x[1,:])), 1)   ,-c_switch,  0  ,  0  ],
                                     [   affinity((torch.linalg.norm(x[0,:] - x[2,:])), 1)   ,  0  ,-c_switch,  0  ],
                                     [   affinity((torch.linalg.norm(x[1,:] - x[2,:])), 1)   ,  0  ,  0  ,-c_switch],
                                   ])
    switchingMatrix *= param

    switchingMatrix00 = -1 * sum(switchingMatrix[:,0])
    switchingMatrix[0][0] = switchingMatrix00

    # CHECK THAT value00 was updated correctly

    
    M = advectionMatrix + diffusionMatrix + switchingMatrix

    # print(diffusionMatrix)

    residue = torch.det(M)
    # print("Determinant of M:", residue)
    # print(M)
    return residue #see what it looks like written out. remember how to take det's of 4x4 matrices


results = []
for i in range(10):
    results.append(getResidue(paths[i,:,:], gradients[i,:,:]).item())
print(results)