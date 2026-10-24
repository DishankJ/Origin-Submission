import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

NX = 3
NU = 2

N = 60
dt = 0.04
Nsim = 250
x_0 = np.array([[0], [0], [0]])

x = ca.SX.sym('x', NX)
u = ca.SX.sym('u', NU)
x_dot = ca.SX.sym("x_dot", 3, 1)
x_dot[0] = u[0]*ca.cos(x[2])
x_dot[1] = u[0]*ca.sin(x[2])
x_dot[2] = u[1]

system = ca.Function('system', [x, u], [x_dot])

ode = {'x': x, 'ode': x_dot, 'p': u}
opts = {'tf': dt}

ode_solver = ca.integrator('F', 'idas', ode, opts)

lb_x = -2*np.ones((NX,1))
lb_x[0] = -2
lb_x[1] = -2
lb_x[2] = -np.pi
lb_x.reshape((1, -1))

ub_x = 2*np.ones((NX,1))
ub_x[0] = 2
ub_x[1] = 2
ub_x[2] = np.pi
ub_x.reshape((1, -1))

lb_u = np.ones((NU,1))
lb_u[0] = -0.26
lb_u[1] = -1.82
ub_u = np.ones((NU,1))
ub_u[0] = 0.26
ub_u[1] = 1.82


Q = np.diag([5, 15, 0.1])
Qf = np.diag([20, 60, 2])
R = np.diag([0.1, 0.1])
# Q = np.diag([2, 25, 0.1])
# Qf = np.diag([8, 35, 5])
# R = np.diag([0.1, 0.1])

x_target1 = np.array([[1.0], [0.05], [0]])
x_target = ca.SX.sym("x_target", NX, 1)

x_diff = x - x_target

J_stage = x_diff.T@Q@x_diff + u.T@R@u
J_terminal = x_diff.T@Qf@x_diff

stage_cost_fcn = ca.Function("J_stage",[x,u, x_target],[J_stage])
terminal_cost_fcn = ca.Function('J_terminal',[x, x_target],[J_terminal])

states = ca.SX.sym('states', NX, N+1)
controls = ca.SX.sym('controls', NU, N)

g = []
lb_g = []
ub_g = []

x_init = ca.SX.sym('x_init', NX)

g.append(states[:, 0] - x_init)
lb_g.append(np.zeros((NX,1)))
ub_g.append(np.zeros((NX,1)))

obs = [0.5, 0.0, 0.2]            # x, y, radius
safe_r = obs[2] + 0.1 + 0.05            # radius enforced by h()

def h(x):
    return (x[0, :] - obs[0])**2 + (x[1, :] - obs[1])**2 - safe_r**2
GAMMA = 0.2

J = 0
for t in range(N):
    g.append(states[:, t+1] - (states[:, t] + dt*system(states[:, t], controls[:, t])))
    lb_g.append(np.zeros((NX,1)))
    ub_g.append(np.zeros((NX,1)))
    g.append(h(states[:, t+1]) - (1-GAMMA)*h(states[:, t]))
    lb_g.append(np.zeros((1,1)))
    ub_g.append(np.full((1,1), np.inf))

    J += stage_cost_fcn(states[:, t], controls[:, t], x_target)
J += terminal_cost_fcn(states[:, N], x_target)

g = ca.vertcat(*g)
prob = {'f': J, 'x': ca.vertcat(ca.reshape(states, NX*(N+1), 1), ca.reshape(controls, NU*N, 1)), 'g':g, 'p': ca.vertcat(x_init, x_target)}
mpc_solver = ca.nlpsol('solver', 'ipopt', prob)

lb_g = ca.vertcat(*lb_g)
ub_g = ca.vertcat(*ub_g)


from casadi.tools import *

opt_x = struct_symSX([
    entry('states', shape=NX, repeat=[N+1]),
    entry('controls', shape=NU, repeat=[N])
])

def detour_guess(x_start, x_goal, side=+1, bulge=safe_r + 0.15):
    s = np.linspace(0, 1, N+1)
    Sg = np.zeros((NX, N+1))
    Sg[0] = x_start[0] + (x_goal[0] - x_start[0])*s
    Sg[1] = x_start[1] + (x_goal[1] - x_start[1])*s + side*bulge*np.sin(np.pi*s)
    Sg[2] = np.arctan2(np.gradient(Sg[1]), np.gradient(Sg[0]))
    return np.concatenate([Sg.T.ravel(), np.zeros(NU*N)])

lb_opt_x = opt_x(0)
ub_opt_x = opt_x(0)

# set desired bounds for input 
lb_opt_x['states'] = lb_x
ub_opt_x['states'] = ub_x

lb_opt_x['controls'] = lb_u
ub_opt_x['controls'] = ub_u

mpc_res = mpc_solver(p=ca.vertcat(x_0, x_target1), lbg=lb_g, ubg=ub_g, lbx = lb_opt_x, ubx = ub_opt_x)


opt_x_k = opt_x(mpc_res['x'])
X_k = np.array([*opt_x_k['states', :, 0]]).reshape(-1)
T1_k = np.array([*opt_x_k['states', :, 1]]).reshape(-1)
T2_k = np.array([*opt_x_k['states', :, 2]]).reshape(-1)

res_x_mpc = [x_0]
res_u_mpc = []
xsoll = x_target1
guess = detour_guess(np.array(x_0).ravel(), np.array(xsoll).ravel())

for i in range(Nsim):
    mpc_res = mpc_solver(p=ca.vertcat(x_0, xsoll), x0=guess,
                         lbg=lb_g, ubg=ub_g, lbx=lb_opt_x, ubx=ub_opt_x)
    sol = np.array(mpc_res['x']).ravel()
    opt_x_k = opt_x(mpc_res['x'])
    u_k = opt_x_k['controls', 0]

    x_next = ode_solver(x0=x_0, p=u_k)['xf']
    x_0 = x_next
    res_x_mpc.append(x_next)
    res_u_mpc.append(u_k)

    # shift the previous solution one step forward as the next initial guess
    S = sol[:NX*(N+1)].reshape(N+1, NX)
    U = sol[NX*(N+1):].reshape(N, NU)
    S = np.vstack([S[1:], S[-1:]])
    U = np.vstack([U[1:], U[-1:]])
    guess = np.concatenate([S.ravel(), U.ravel()])


#plots   
res_x_mpc = np.concatenate(res_x_mpc, axis=1)
res_u_mpc = np.concatenate(res_u_mpc, axis=1)

plt.plot([dt*i for i in range(len(res_x_mpc[0,:]))], res_x_mpc[0,:])
plt.plot([dt*i for i in range(len(res_x_mpc[1,:]))], res_x_mpc[1,:])
plt.plot([dt*i for i in range(len(res_x_mpc[2,:]))], res_x_mpc[2,:])
plt.grid()
plt.legend(['x (m)', 'y (m)', 'theta (rad)'])
plt.show()
plt.plot([dt*i for i in range(res_u_mpc.shape[1])], res_u_mpc.T)
plt.grid()
plt.legend(['v (m/s)', 'omega (rad/s)'])
plt.show()

from matplotlib.patches import Polygon

def robot_triangle(x, y, theta, size=0.03, **kwargs):
    """Triangle centered at (x, y), with its tip pointing along theta."""
    # Triangle in the robot frame: tip at the front, two corners at the back
    pts = np.array([[ size,      0.0],
                    [-size*0.6,  size*0.6],
                    [-size*0.6, -size*0.6]])
    R_mat = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
    pts = pts @ R_mat.T + np.array([x, y])
    return Polygon(pts, closed=True, **kwargs)

fig, ax = plt.subplots(figsize=(7, 7))

# Path
ax.plot(res_x_mpc[0, :], res_x_mpc[1, :], 'b-', lw=1.5, label='path')

# Triangles along the path (every `step` samples) to show heading
step = 5
for k in range(0, res_x_mpc.shape[1], step):
    ax.add_patch(robot_triangle(res_x_mpc[0, k], res_x_mpc[1, k], res_x_mpc[2, k],
                                facecolor='lightblue', edgecolor='blue', alpha=0.7))

# Start (green) and final (red) pose
ax.add_patch(robot_triangle(res_x_mpc[0, 0], res_x_mpc[1, 0], res_x_mpc[2, 0],
                            facecolor='green', edgecolor='k', label='start'))
ax.add_patch(robot_triangle(res_x_mpc[0, -1], res_x_mpc[1, -1], res_x_mpc[2, -1],
                            facecolor='red', edgecolor='k', label='final'))

# Target pose
ax.plot(x_target1[0], x_target1[1], 'k*', ms=14, label='target')
ax.add_patch(robot_triangle(x_target1[0, 0], x_target1[1, 0], x_target1[2, 0],
                            facecolor='none', edgecolor='k', ls='--'))
ax.add_patch(plt.Circle((obs[0], obs[1]), obs[2], color='r', alpha=0.4, label='obstacle'))
ax.add_patch(plt.Circle((obs[0], obs[1]), safe_r, fill=False, ls='--', color='r', label='safety radius'))
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('MPC path of TurtleBot')
ax.set_aspect('equal', adjustable='datalim')   # keeps triangles undistorted
ax.grid(True)
ax.legend()
plt.show()