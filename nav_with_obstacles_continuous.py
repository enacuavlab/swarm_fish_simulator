import sys

from PyQt6.QtWidgets import QApplication

from swarm_controller_simple_sim import SwarmFish_Environment, SwamFish_View, SwarmFish_Controller, ms_of_hz, make_args_parser

import numpy as np
import math

import swarmfish.swarm_control as sc
import swarmfish.obstacles as so
from swarmfish.utils import load_polygons_from_json

import panel_flow as pgflow
from building import Building

SHOW_DIRECTION = True
SHOW_INFLUENTIALS = False
USE_PANEL_FLOW = True
NB_INFLUENTIAL = 1
SCENE_FILE = "scenes/test_world.json"

POS_NOISE = 0.
SPEED_NOISE = 0. #0.1
HEADING_NOISE = 0.

class SwarmFish_Scenario(SwarmFish_Controller):

    def __init__(self, ARGS, env, view, start=True):
        super().__init__(ARGS, env, view)

        #### Init SwarmFish ########################################
        arena_radius = 10.
        arena_center = np.array([0., 0., 0.])
        self.arena = so.Arena(center=arena_center[0:2], radius=arena_radius, name="arena")

        # larger grid
        self.view.grid.setSize(100,100,1)
        self.view.grid.setSpacing(10,10,1)
        # load obstacles from json
        self.buildings = load_polygons_from_json(SCENE_FILE)
        for b in self.buildings:
            self.view.add_polygon(vertices=b.vertices, height=b.z_max-b.z_min)
        # nav waypoints
        self.wps = np.array([
            [-10., 50., 5.],
            [10, -50., 5.]
            ])
        self.wps_idx = 0
        self.wps_nb = np.shape(self.wps)[0]
        # panel flow
        if USE_PANEL_FLOW:
            self.pgflow_buildings = []
            self.pgflow_vehicles = []
            for b in self.buildings:
                vert = b.get_top()
                self.pgflow_buildings.append(Building(b.get_top()))
            for d in range(self.num_drones):
                pos = self.obs[d].pos
                goal = self.wps[self.wps_idx]
                vehicle = pgflow.Vehicle(ID=d, position=pos, goal=goal)
                self.pgflow_vehicles.append(vehicle)
            self.pgflow = pgflow.PanelFlow()
            self.pgflow.prepare_buildings(self.pgflow_buildings, panel_size=1.)

        if SHOW_INFLUENTIALS:
            self.lines = {}
            for d in range(self.num_drones):
                self.lines[d] = [ self.view.add_line(np.zeros(3), np.zeros(3)) for x in range(NB_INFLUENTIAL) ]

        if SHOW_DIRECTION:
            self.directions = {}
            self.speeds = {}
            self.flow_lines = {}
            for d in range(self.num_drones):
                self.directions[d] = self.view.add_line(np.zeros(3), np.zeros(3), color=(1.,0.,0.,1.))
                self.speeds[d] = self.view.add_line(np.zeros(3), np.zeros(3), color=(0.,0.,1.,1.))
                self.flow_lines[d] = self.view.add_line(np.zeros(3), np.zeros(3), color=(0.,1.,0.,1.))

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
            uav_name = str(uav_id)
            state = states[uav_id]
            if USE_PANEL_FLOW:
                self.pgflow_vehicles[uav_id].position = state.pos
                self.flow = self.pgflow.Flow_Velocity_Calculation(self.pgflow_vehicles[uav_id], self.pgflow_buildings)
                nav_direction = math.atan2(self.flow[1], self.flow[0])
                if math.isnan(nav_direction):
                    nav_direction = None
                #print(state.pos, self.flow, nav_direction)
                #print(uav_id, flow, f'{np.degrees(nav_direction):.2f}')
            else:
                dpos = self.wps[self.wps_idx][0:2] - state.pos[0:2]
                nav_direction = math.atan2(dpos[1], dpos[0])
                self.flow = np.array([math.cos(nav_direction), math.sin(nav_direction)]) # for display
            nav_alt = self.wps[self.wps_idx][2]
            obstacles = [ b.get_wall(state, self.params) for b in self.buildings if b is not None ]
            if uav_id in self.intruders_id:
                cmd, _ = sc.compute_interactions(state, self.params, [], nb_influent=0, 
                        direction=self.direction_setpoint,
                        speed=self.speed_setpoint*2., altitude=self.altitude_setpoint, z_min=1., z_max=10., obstacles=obstacles)
            else:
                neighbors = [ states[k] for k in range(self.num_drones) if (k != uav_id) and (k not in self.intruders_id)  ]
                intruders = [ states[k] for k in self.intruders_id if k != uav_id ]
                cmd, influentials = sc.compute_interactions(state, self.params, neighbors, nb_influent=NB_INFLUENTIAL,
                        speed=self.speed_setpoint, altitude=nav_alt, z_min = 1., z_max = 10.,
                        direction=nav_direction,
                        intruders=intruders, obstacles=obstacles)
                if SHOW_INFLUENTIALS:
                    for l, influential in zip(self.lines[uav_id], influentials):
                        self.view.move_line(l, state.pos, states[influential[1]].pos)
            dt = 1. / self.control_freq_hz
            speed_cmd = state.get_speed_2d() + cmd.a_bx * dt
            vz_cmd = state.get_vz() + cmd.a_z * dt
            yaw_cmd = cmd.a_by / max(state.get_speed_2d(),0.1)
            speed = np.array([
                speed_cmd * math.cos(state.get_course(use_heading=True)),
                speed_cmd * math.sin(state.get_course(use_heading=True)),
                vz_cmd,
                yaw_cmd])
            self.commands[uav_id] = speed

            if SHOW_DIRECTION:
                self.view.move_line(self.directions[uav_id], state.pos, state.pos+speed[0:3])
                self.view.move_line(self.speeds[uav_id], state.pos, state.pos+state.speed)
                self.view.move_line(self.flow_lines[uav_id], state.pos, state.pos+np.hstack((self.flow, np.array([0.]))))

        # test waypoint reached
        dwps = [np.linalg.norm(self.wps[self.wps_idx][0:2] - states[i].pos[0:2]) for i in range(self.num_drones)]
        if np.min(dwps) < 10.:
            self.wps_idx = (self.wps_idx + 1) % self.wps_nb
            if USE_PANEL_FLOW:
                goal = self.wps[self.wps_idx]
                for uav_id in range(self.num_drones):
                    self.pgflow_vehicles[uav_id].goal = goal


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
