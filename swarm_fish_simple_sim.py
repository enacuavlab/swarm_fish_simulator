import sys

from PyQt6.QtWidgets import QApplication

from swarm_controller_simple_sim import SwarmFish_Environment, SwamFish_View, SwarmFish_Controller, ms_of_hz, make_args_parser

import numpy as np
import math

import swarmfish.swarm_control as sc
import swarmfish.obstacles as so

TEST_OBSTACLE = False
SHOW_DIRECTION = True
SHOW_ARENA = True
SHOW_INFLUENTIALS = False
NB_INFLUENTIAL = 2

POS_NOISE = 0.
SPEED_NOISE = 0. #0.1
HEADING_NOISE = 0.

class SwarmFish_Scenario(SwarmFish_Controller):

    def __init__(self, ARGS, env, view, start=True):
        super().__init__(ARGS, env, view)

        #### Init SwarmFish ########################################
        arena_radius = 25.
        arena_center = np.array([0., 0., 0.])
        self.arena = so.Arena(center=arena_center[0:2], radius=arena_radius, name="arena")
        if SHOW_ARENA:
            #self.view.add_cylinder(radius=arena_radius, height=0.01, pos=arena_center, color=(0,1,0,1))
            self.view.add_circle(radius=arena_radius, pos=arena_center, color=(0,1,0,1))

        init_yaw = [ self.obs[j].att[2] for j in range(self.num_drones) ]
        #self.desired_course = np.array(init_yaw) # np.zeros(self.num_drones)

        if TEST_OBSTACLE:
            obstacle_radius = 0.5
            obstacle_center = np.array([0, 4., 0])
            obstacle_z_min = 0.
            obstacle_z_max = 4.
            self.obstacle = so.CircleObstacle(obstacle_center[0:2], obstacle_radius, obstacle_z_min, obstacle_z_max, name='pole')
            self.view.add_cylinder(radius=obstacle_radius,
                    height=obstacle_z_max-obstacle_z_min,
                    pos=obstacle_center)
            polygon_vertices = np.array([[1., -1.], [1., 1.], [-2., 1.], [-1., -1.]]) + np.full((4,2),np.array([-3., -4.]))
            self.polygon = so.PolygonObstacle(polygon_vertices, obstacle_z_min, obstacle_z_max, name="polygon")
            self.view.add_polygon(vertices=polygon_vertices, height=obstacle_z_max-obstacle_z_min)

        if SHOW_INFLUENTIALS:
            self.lines = {}
            for d in range(env.num_drones):
                self.lines[d] = [ self.view.add_line(np.zeros(3), np.zeros(3)) for x in range(NB_INFLUENTIAL) ]

        if SHOW_DIRECTION:
            self.directions = {}
            self.speeds = {}
            for d in range(env.num_drones):
                self.directions[d] = [ self.view.add_line(np.zeros(3), np.zeros(3), color=(1.,0.,0.,1.)) ]
                self.speeds[d] = [ self.view.add_line(np.zeros(3), np.zeros(3), color=(0.,0.,1.,1.)) ]

        if start:
            self.start_simulation()


    def update_action(self):
        #### Step the simulation ###################################

        noise = np.random.normal(size=7)
        states = { j: sc.State(
            self.obs[j].pos + POS_NOISE*noise[0:3],
            self.obs[j].vel + SPEED_NOISE*noise[3:6],
            self.obs[j].att[2] + HEADING_NOISE*noise[6],
            self.current_time, str(j)) for j in range(self.num_drones) }
        #### Compute control for the current state #############
        for uav_id in range(self.num_drones):
            # If you need : obs[str(j)]["state"] include jth vehicle's
            # position [0:3]  quaternion [3:7]   Attitude[7:10]  VelocityInertialFrame[10:13]     qpr[13:16]   motors[16:20]
            #   X  Y  Z       Q1   Q2   Q3  Q4   R  P   Y        VX     VY    VZ                  WX WY WZ     P0 P1 P2 P3
            uav_name = str(uav_id)
            state = states[uav_id]
            wall = self.arena.get_wall(state, self.params)
            obstacles = []
            if TEST_OBSTACLE:
                obstacles = [ o.get_wall(state, self.params) for o in [self.obstacle, self.polygon] if o is not None ]
            if uav_id in self.intruders_id:
                cmd, _ = sc.compute_interactions(state, self.params, [], nb_influent=0,
                        wall=wall, altitude=5., z_min=1., z_max=10., obstacles=obstacles)
            else:
                neighbors = [ states[k] for k in range(self.num_drones) if (k != uav_id) and (k not in self.intruders_id)  ]
                intruders = [ states[k] for k in self.intruders_id if k != uav_id ]
                cmd, influentials = sc.compute_interactions(state, self.params, neighbors, nb_influent=NB_INFLUENTIAL, wall=wall,
                        altitude=self.altitude_setpoint, z_min = 1., z_max = 10.,
                        direction=self.direction_setpoint,
                        intruders=intruders, obstacles=obstacles)
                if SHOW_INFLUENTIALS:
                    for l, influential in zip(self.lines[uav_id], influentials):
                        self.view.move_line(l, state.pos, states[int(influential[1])].pos)
            #desired_course = sc.wrap_to_pi(state.get_course() + cmd.delta_course)
            #self.desired_course[uav_id] = sc.wrap_to_pi(self.desired_course[uav_id] + cmd.delta_course / self.simulation_freq_hz)
            #desired_course = self.desired_course[uav_id] + 0.1*np.random.rand()
            yaw_rate = cmd.delta_course #/ self.simulation_freq_hz
            #print(uav_id, yaw_rate)
            #print(f'desired course {uav_name}: {np.degrees(desired_course):0.2f} | {np.degrees(state.get_course()):0.2f} + {np.degrees(cmd.delta_course):0.2f}')
            # TODO fix heading/rates
            magnitude = self.speed_setpoint + cmd.delta_speed # TODO clip min/max
            if uav_id in self.intruders_id:
                magnitude *= 2.
            speed = np.array([
                magnitude * math.cos(state.get_course(use_heading=True)),
                magnitude * math.sin(state.get_course(use_heading=True)),
                cmd.delta_vz,
                yaw_rate])
            self.commands[uav_id] = speed

            if SHOW_DIRECTION:
                for d, s in zip(self.directions[uav_id], self.speeds[uav_id]):
                    self.view.move_line(d, state.pos, state.pos+speed[0:3])
                    self.view.move_line(s, state.pos, state.pos+state.speed)

            #print(state)
            #print(f' wall {uav_id} | dist {wall[0]:.2f}, angle= {np.degrees(wall[1]):.2f}')
            #print(f' cmd {uav_id} | {np.degrees(cmd.delta_course):0.2f}, {cmd.delta_speed:0.3f}, {cmd.delta_vz:0.3f} | desired_course {np.degrees(desired_course):0.2f}')
            #print(' speed',uav_id,speed)
        #print('') # blank line


if __name__ == "__main__":

    parser = make_args_parser()
    args = parser.parse_args()
    env = SwarmFish_Environment(args)
    app = QApplication(sys.argv)
    view = SwamFish_View()
    controller = SwarmFish_Scenario(args, env, view)
    app.exec()
    controller.close()
    view.close()
    env.close()
    sys.exit()
