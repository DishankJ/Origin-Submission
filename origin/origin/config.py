import numpy as np

chosenTrajectory = 'trapezoidal'
visualiseTrajectory = True

N = 15
dt = 0.04

v_max = 0.26
w_max = 1.82
a_max = 0.2

NX = 3
NU = 2

waypoints = np.array([
                    [0.29,  0.0],
                    [1.2,  0.0],
                    [1.435, 0.914],
                    [1.9, 1.1],
                    ])

waypoints2 = np.array([
                    [1.0,  0.25],
                    [2.0,  0.0],   # target
                    ])

Q = np.diag([30, 45, 0.05])
Qf = np.diag([70, 80, 0.1])
R = np.diag([0.1, 0.1])

Q2 = np.diag([40, 40, 0.01])
Qf2 = np.diag([60, 60, 0.01])
R2 = np.diag([0.1, 0.1])

GAMMA = 0.3

obstacle_rad = 0.3
obstacles = [[1.0, 0.0, obstacle_rad]]