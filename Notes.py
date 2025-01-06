"""
Overall TO-DO
-------------------
Calculate what would happen by hand when switching parameters in a certain way and see what happens in the code with that. Mess mainly with alpha, epsilon, etc
Make a table of important parameters and their associated values


Next TO-DO for Code
-------------------
1.Make Github

2.
Qualitative comparison matlab vs python. Then, if feeling confident, what measures could we do to check that values are same across python matlab. 
first and foremost check moments for formtimes, breaktimes, and brownian. Strike a balance between time and impact in terms of coding ease/worthwhileness.
compare seeded run(s) in matlab vs python?

ML
Determinant function works? see if that works in a copy of the 1-bead code!!!

Two beads binding and never unbinding?
Equilateral triangle is too large?
Seems like binding times are off, answer why?
understand that you SEED globally, THEN GENERATE seaprately
rng is deterministic once seeded.

DONE
-------------------
Fix seeding
Check that plots work for perfect squares
Update code to output a class instance that has positions, states
Update class to include seed, simulation parameters, optionalPrintStatements
Update class to include sum of (forces, noise, and ctmc value) at each time step
?Update graphs to include seeds?
Turn off crosslinking force and see what happens with the goal of the position graph reconciling with the bonds' formation/breaking
Code up R = det[M], where M = advection + diffusion + Switching
[43]
[12]

[21]
[34]x

diagonals flip 01 and 10. 
Class Attributes
----------------
positions
states
seed
simulation parameters (all probably,dynamic only)
nonCTMC forces (potential confinement + exclcuded volume + drift)
ctmc forces (stochastic crosslinking)

Method that prints its info e.g. this is {self.numSims} simulations of a {self.p} bead, {self.d}-dimensional 
 system with crucial parameters {self.dynamicParams}. Seeds are {self.seeds[i] for i in range(numSims)}

























    # Can improve speed by only calculating formTimes if the section utilizing them is reached.
    #  so create a variable called formTimesCalculated and condition calculation on if that variable is equal to 0 and 
    # change it to 1 once calculated.

    # Also have already sped it up by not calculating the formtime when considering the same bead to itself (ignored CTMC diagonal)


# Number of instances = sum(range(p)) EQUALS sum from 0 to p-1 EQUALS p*(p-1)/2 EQUALS size of upper triangle of a pxp matrix

# Optimize speed by storing formation times in a minHeap rather than an array

            # # above is effectively the same as writing
            # if (formTimes[0] < dt and
            #     k in formTimes[0] and 
            #     (lambda x: 3 - x)(formTimes[0].index(k)) not in visitedBeads and
            #     state[(lambda x: 3 - x)(formTimes[0].index(k))] == (lambda x: 3 - x)(formTimes[0].index(k))):

"""