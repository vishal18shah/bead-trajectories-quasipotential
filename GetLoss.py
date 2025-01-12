import numpy as np
import torch
import sympy as sp
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
    #both x(positions) and gradU(gradient of potential wrt positions) should be 3x2 tensors. in the general case, they are p by d in shape
    v1 = torch.zeros(3, 2)
    for k in range(3):
        v1[k, :] += -1 * k_c * x[k,:] * (torch.norm(x[k,:]) ** 2)
        for j in range(3):
            if j == k: continue
            v1[k, :] += a_ev * (x[k,:] - x[j,:]) * torch.exp(-1 * (torch.norm(x[k,:] - x[j,:]) ** 2) / c_ev) 

    v2 = v1 + torch.stack([
    k_v * (x[0, :] - x[1, :]),
    k_v * (x[1, :] - x[0, :]),
    torch.tensor([0.0, 0.0])
])

    v3 = v1 + torch.stack([
        k_v * (x[0, :] - x[2, :]),
        torch.tensor([0.0, 0.0]),
        k_v * (x[2, :] - x[0, :])
    ])

    v4 = v1 + torch.stack([
        torch.tensor([0.0, 0.0]),
        k_v * (x[1, :] - x[2, :]),
        k_v * (x[2, :] - x[1, :])
    ])

    
    advectionForces = [v1, v2, v3, v4]
    advectionMatrix = torch.zeros(4, 4)
    diffusionMatrix = torch.zeros(4, 4)

    for i in range(4):
        advectionMatrix[i,i] = torch.sum(advectionForces[i] * gradU) # * does element-wise multiplication
        diffusionMatrix[i,i] = torch.sum(gradU * gradU)


    # make sure dimensionality is correct for torch.linalg.norm
    switchingMatrix = torch.tensor([ [                       0                             ,c_switch,c_switch ,c_switch ],
                                     [   affinity((torch.linalg.norm(x[0,:] - x[1,:])), 1)   ,-c_switch,  0  ,  0  ],
                                     [   affinity((torch.linalg.norm(x[0,:] - x[2,:])), 1)   ,  0  ,-c_switch,  0  ],
                                     [   affinity((torch.linalg.norm(x[1,:] - x[2,:])), 1)   ,  0  ,  0  ,-c_switch],
                                   ])
    switchingMatrix *= param

    switchingMatrix00 = -1 * sum(switchingMatrix[0, :])
    switchingMatrix[0][0] = switchingMatrix00
    # CHECK THAT value00 was updated correctly

    
    M = advectionMatrix + diffusionMatrix + switchingMatrix
    residue = torch.det(M)
    print("Determinant of M:", residue)
    return residue #see what it looks like written out. remember how to take det's of 4x4 matrices
   

def getLoss(positions): #One simulation at a time so positions has shape (p,d,nt+1).
    loss = torch.mean(np.array([getResidue(positions[:, :, t]) for t in range(positions.shape[2])]) ** 2 )
    return None


# def testGetResidue():
#     x1, y1, x2, y2, x3, y3 = sp.symbols('x1 y1 x2 y2 x3 y3')
#     X = sp.Matrix([
#         [x1, y1],
#         [x2, y2],
#         [x3, y3]
#     ])

#     w1, w2, w3, w4, w5, w6 = sp.symbols('w1 w2 w3 w4 w5 w6')
#     W = sp.Matrix([
#         [w1, w2],
#         [w3, w4],
#         [w5, w6]
#     ])

#     getResidue(X,W)

# testGetResidue()

# x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], requires_grad=True)  # Shape: (3, 2)
# print(x[2, :])
# gradU = torch.tensor([[0.5, 0.5], [-0.5, -0.5], [0.2, -0.2]])  # Shape: (3, 2)
# residue_value = getResidue(x, gradU)
quasipotentials = [0,0.000756157630320576,0.0023063877634836,0.00392045596608979,0.00557131184207416,0.0072124749156677,0.00877898352780071,0.0100886275944653,0.0108680776679456,0.0110942353446692]
positions = [0,-3.97633854000509e-07,9.0552639727391e-08,-3.40715433050964e-07,-5.47292092959352e-09,-4.62393704440715e-07,9.90457082337456e-08,-5.40888909049959e-07,2.5347389813985e-07,-6.88432291351422e-07
,-0.52335374890531,-0.501733682143641,-0.478707197878456,-0.4568210759134,-0.433459894240393,-0.411557421393789,-0.388050459367647,-0.371993867888114,-0.355715767422683,-0.332405611334738
,0,-5.58714812466072e-07,2.69065391002746e-07,-2.75070992100107e-07,-1.87505058237965e-07,-6.61021338295601e-07,9.37979751620383e-08,-6.81089464805657e-07,2.8929976086067e-07,-8.57864621588462e-07
,-0.52335374890531,-0.522715253344699,-0.522733547661897,-0.52285931259943,-0.523238946332131,-0.523236533258431,-0.523059699924465,-0.512634919655211,-0.495504829322485,-0.493180216003928
,0,2.57906198284196e-26,-1.29041243589183e-26,0,0,0,0,0,0,0,0.65938440479222,0.636364805947243,0.614742595397165,0.591967216812252,0.570710523466624,0.547950550715655,0.526858684480272,0.501735308907261,0.480778084516878,0.459589023056138]

print(len(quasipotentials))