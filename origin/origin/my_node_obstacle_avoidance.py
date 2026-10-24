import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from math import atan2

import casadi as ca
from casadi.tools import *
import numpy as np
from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import origin.config as config


def generateSmoothTrajectory(waypoints, trajectoryType: str, visualiseTrajectory: bool):
    p = np.asarray(waypoints, dtype=float)
    if not np.array_equal(p[0], [0,0]):
        p = np.insert(p, 0, [0,0], axis=0)
    lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)

    if len(p) < 2:
        assert len(p) > 0, "Give at least one waypoint"
        return [(p[0, 0], p[0, 1], 0.0)]

    d = np.r_[0, np.cumsum(lengths)]
    sx = CubicSpline(d, p[:, 0])
    sy = CubicSpline(d, p[:, 1])
    u = np.linspace(0, d[-1], 1000)
    curve = np.c_[sx(u), sy(u)]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    L = arc[-1]

    if trajectoryType == 'trapezoidal':
        ta = min(config.v_max / config.a_max, np.sqrt(L / config.a_max))
        vp = config.a_max * ta
        da = 0.5 * config.a_max * ta**2
        tc = max(0, (L - 2 * da) / vp)
        T = 2 * ta + tc

        t = np.r_[np.arange(0, T, config.dt), T]
        s = np.where(t < ta, 0.5 * config.a_max * t**2,
            np.where(t < ta + tc, da + vp * (t - ta),
                    L - 0.5 * config.a_max * (T - t)**2))
    elif trajectoryType == 'constantvelocity':
        T = L / config.v_max
        t = np.r_[np.arange(0, T, config.dt), T]
        s = config.v_max * t
    else:
        raise ValueError("Chosen trajectory generation is not possible")
    u_t = np.interp(s, arc, u)
    smoothTrajectory = list(zip(sx(u_t), sy(u_t), t))
    if visualiseTrajectory:
        plt.plot([item[1] for item in smoothTrajectory], [item[0] for item in smoothTrajectory], 'o', markersize=2)
        plt.plot(waypoints[:, 1], waypoints[:, 0], 'o', markersize=5)
        plt.show()
    return smoothTrajectory

class MyNode(Node):
    def __init__(self):
        super().__init__("my_node")
        self.pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.timer = self.create_timer(config.dt, self.publish_control)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.create_subscription(TwistStamped, "/cmd_vel", self.vel_callback, 10)

        self.x_0 = np.array([[0], [0], [0]])
        self.trajectoryType = config.chosenTrajectory
        self.visualiseTrajectory = config.visualiseTrajectory
        # CHANGED: reference goes straight THROUGH the obstacle zone, passing (1.0, 0.25) which is
        # inside the radius 0.3 -> the reference is not collision free, so the CBF has to bend the path
        self.waypoints = config.waypoints2
        self.smooth_trajectory = generateSmoothTrajectory(self.waypoints, self.trajectoryType, self.visualiseTrajectory)
        self.obstacles = config.obstacles   # CHANGED: obstacle at (1.0, 0)
        self.current_position_odom = np.array([[0], [0], [0]])
        self.res_x_mpc = [self.current_position_odom]
        self.res_u_mpc = []

        traj = np.array([(x, y) for x, y, _ in self.smooth_trajectory])
        plt.ion()
        self.fig, ax = plt.subplots()
        ax.plot(traj[:, 0], traj[:, 1], 'b--', label='reference')

        # CHANGED: only the real CBF boundary (radius r) is drawn
        for i, (ox, oy, r) in enumerate(self.obstacles):
            ax.add_patch(Circle((ox, oy), r, color='r', alpha=0.4,
                                label='obstacle (CBF boundary)' if i == 0 else None))

        # CHANGED: trail of the actual robot path, so the CBF turning maneuver is visible
        self.trail_x, self.trail_y = [], []
        self.trail, = ax.plot([], [], 'g-', lw=2, label='actual path')

        self.robot, = ax.plot([], [], 'k', ls='', marker=(3, 0, 0), markersize=15)
        ax.set_aspect('equal')
        ax.autoscale_view()
        ax.grid(True)
        ax.legend()
        self.create_timer(0.1, self.update_plot)

        self.setup_mpc()
        self.timeIndex = 0

    def update_plot(self):
        x, y, th = np.asarray(self.current_position_odom, dtype=float).flatten()
        self.robot.set_data([x], [y])
        self.robot.set_marker((3, 0, np.degrees(th) - 90))
        self.trail_x.append(x)
        self.trail_y.append(y)
        self.trail.set_data(self.trail_x, self.trail_y)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def setup_mpc(self,):
        x = ca.SX.sym('x', config.NX)
        u = ca.SX.sym('u', config.NU)
        x_dot = ca.SX.sym("x_dot", 3, 1)
        x_dot[0] = u[0]*ca.cos(x[2])
        x_dot[1] = u[0]*ca.sin(x[2])
        x_dot[2] = u[1]

        system = ca.Function('system', [x, u], [x_dot])

        ode = {'x': x, 'ode': x_dot, 'p': u}
        opts = {'tf': config.dt}

        self.ode_solver = ca.integrator('F', 'idas', ode, opts)

        lb_x = -4*np.ones((config.NX,1))
        lb_x[0] = -4
        lb_x[1] = -4
        lb_x[2] = -np.pi
        lb_x.reshape((1, -1))

        ub_x = 4*np.ones((config.NX,1))
        ub_x[0] = 4
        ub_x[1] = 4
        ub_x[2] = np.pi
        ub_x.reshape((1, -1))

        lb_u = np.ones((config.NU,1))
        lb_u[0] = -config.v_max
        lb_u[1] = -config.w_max
        ub_u = np.ones((config.NU,1))
        ub_u[0] = config.v_max
        ub_u[1] = config.w_max

        x_target = ca.SX.sym("x_target", config.NX, 1)

        x_diff = x - x_target

        J_stage = x_diff.T@config.Q2@x_diff + u.T@config.R2@u
        J_terminal = x_diff.T@config.Qf2@x_diff

        stage_cost_fcn = ca.Function("J_stage",[x,u, x_target],[J_stage])
        terminal_cost_fcn = ca.Function('J_terminal',[x, x_target],[J_terminal])

        states = ca.SX.sym('states', config.NX, config.N+1)
        controls = ca.SX.sym('controls', config.NU, config.N)

        g = []
        lb_g = []
        ub_g = []

        x_init = ca.SX.sym('x_init', config.NX)

        g.append(states[:, 0] - x_init)
        lb_g.append(np.zeros((config.NX,1)))
        ub_g.append(np.zeros((config.NX,1)))

        J = 0
        for t in range(config.N):
            g.append(states[:, t+1] - (states[:, t] + config.dt*system(states[:, t], controls[:, t])))
            lb_g.append(np.zeros((config.NX,1)))
            ub_g.append(np.zeros((config.NX,1)))
            # CBF constraint
            for obs in self.obstacles:
                g.append(self.h(states[:, t+1], obs) - (1-config.GAMMA)*self.h(states[:, t], obs))
                lb_g.append(np.zeros((1,1)))
                ub_g.append(np.full((1,1), np.inf))

            J += stage_cost_fcn(states[:, t], controls[:, t], x_target)
        J += terminal_cost_fcn(states[:, config.N], x_target)

        g = ca.vertcat(*g)
        prob = {'f': J, 'x': ca.vertcat(ca.reshape(states, config.NX*(config.N+1), 1), ca.reshape(controls, config.NU*config.N, 1)), 'g':g, 'p': ca.vertcat(x_init, x_target)}
        self.mpc_solver = ca.nlpsol('solver', 'ipopt', prob)

        self.lb_g = ca.vertcat(*lb_g)
        self.ub_g = ca.vertcat(*ub_g)

        self.opt_x = struct_symSX([
            entry('states', shape=config.NX, repeat=[config.N+1]),
            entry('controls', shape=config.NU, repeat=[config.N])
        ])

        self.lb_opt_x = self.opt_x(0)
        self.ub_opt_x = self.opt_x(0)

        # set desired bounds for input 
        self.lb_opt_x['states'] = lb_x
        self.ub_opt_x['states'] = ub_x

        self.lb_opt_x['controls'] = lb_u
        self.ub_opt_x['controls'] = ub_u

    def h(self, x, obs):
        x_obs, y_obs, r_obs = obs
        h = (x[0] - x_obs)**2 + (x[1] - y_obs)**2 - (r_obs)**2
        return h
        
    def publish_control(self):
        self.x_0 = self.current_position_odom
        target = np.array([[x, y, 0.0] for (x, y, t) in self.smooth_trajectory])[min(self.timeIndex, len(self.smooth_trajectory) - 1)]
        # CHANGED: initial guess = "stay where I am" (default guess is all zeros, which makes the solver
        # get stuck at the obstacle boundary)
        guess = ca.vertcat(np.tile(np.asarray(self.x_0, dtype=float), (config.N+1, 1)), np.zeros((config.NU*config.N, 1)))
        mpc_res = self.mpc_solver(x0=guess, p=ca.vertcat(self.x_0, target), lbg=self.lb_g, ubg=self.ub_g, lbx=self.lb_opt_x, ubx=self.ub_opt_x)
        opt_x_k = self.opt_x(mpc_res['x'])
        u_k = opt_x_k['controls', 0]
 
        res_integrator = self.ode_solver(x0=self.x_0, p=u_k)
        x_next = res_integrator['xf']
    
        self.current_position = x_next
        self.res_x_mpc.append(self.x_0)
        self.res_u_mpc.append(u_k)
 
        # CHANGED: log the barrier value (h ~ 0 means the robot is on the CBF boundary, i.e. CBF is active)
        h_now = float(self.h(np.asarray(self.x_0, dtype=float).flatten(), self.obstacles[0]))
        self.get_logger().info("Odom pos " + str(self.current_position_odom.flatten()) + ", h = " + str(round(h_now, 4)))
        msg = TwistStamped()
        msg.twist.linear.x = float(u_k[0])
        msg.twist.angular.z = float(u_k[1])
        self.pub.publish(msg)

        self.timeIndex += 1

    def vel_callback(self, msg: TwistStamped):
        self.get_logger().info("speed is " + str(msg.twist.linear.x) + ", ang speed is " + str(msg.twist.angular.z))

    def odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        theta = atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.current_position_odom = np.array([[msg.pose.pose.position.x],
                                               [msg.pose.pose.position.y],
                                               [theta]])

def main(args=None):
    rclpy.init(args=args)

    node = MyNode()
    rclpy.spin(node)
    
    rclpy.shutdown()