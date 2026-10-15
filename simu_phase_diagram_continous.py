"""Script for testing swarmfish with quadrotors following a desired velocity vector (unit vector + magnitude normalized by max vehicle speed in Km/h)
"""
import sys
import os
import argparse
import time
from datetime import datetime
import numpy as np
import math

from simple_sim.point_sim import PointSim, QuadrotorState
from swarm_controller_simple_sim import SwarmFish_Environment, ms_of_hz, make_args_parser
from utils.util_logger import Logger

from swarmfish.utils import load_params_from_yaml, compute_quantification
import swarmfish.swarm_control as sc


def run_simulation(ARGS: dict):
    #### Initialize the simulation #############################
    time_step = 1. / float(ARGS.simulation_freq_hz)
    dim = math.ceil(math.sqrt(ARGS.num_drones))
    dist = 2.

    init_state = np.array([[ dist*(i % dim), dist*math.floor(i / dim), 0., 0. ] for i in range(ARGS.num_drones)]) 
    if ARGS.random_init:
        init_state[:,3] = 2.*np.pi*np.random.rand(ARGS.num_drones)

    NOISES = [ np.radians(0.2), 0.01, 0.05 ] # noise on heading, speed and vertical speed (standard deviation of a normal distribution)

    #### Create the environment
    sim = [ PointSim(init_position=x0[0:3], init_yaw=x0[3]) for x0 in init_state ]

    def get_states():
        return [ sim[i].get_state() for i in range(ARGS.num_drones) ]

    def step(commands):
        for i in range(ARGS.num_drones):
            sim[i].update_state(
                    desired_velocity=commands[i][0:3],
                    desired_yaw_rate=commands[i][3],
                    time_step=time_step)
        return get_states()

    #### Initialize the logger #################################
    logger = Logger(
        logging_freq_hz=int(ARGS.log_freq_hz),
        num_drones=ARGS.num_drones,
        duration_sec=int(ARGS.duration_sec),
        control_length=4
    )
    LOG_EVERY_N_STEPS = int(np.floor(ARGS.simulation_freq_hz / ARGS.log_freq_hz))

    #### Initialize the controllers ############################
    commands = [ np.zeros(4) for i in range(ARGS.num_drones) ]
    speed_setpoint = ARGS.speed_setpoint
    obs = step(commands) # initial observation

    #### Init SwarmFish ########################################
    params = load_params_from_yaml(ARGS.swarm_config)
    if ARGS.y_ali is not None:
        params.y_ali = ARGS.y_ali
    if ARGS.y_att is not None:
        params.y_att = ARGS.y_att

    #### Run the simulation ####################################
    START = time.time()
    for i in range(0, int(ARGS.duration_sec * ARGS.simulation_freq_hz)):

        #### Step the simulation ###############################
        obs = step(commands)

        states = { j: sc.State(
            obs[j].pos,
            obs[j].vel,
            obs[j].att[2],
            float(i / ARGS.simulation_freq_hz))
            for j in range(ARGS.num_drones) }
        #### Compute control for the current state #############
        for uav_id in range(ARGS.num_drones):
            uav_name = str(uav_id)
            state = states[uav_id]
            neighbors = [ states[k] for k in range(ARGS.num_drones) if k != uav_id ]
            cmd, influentials = sc.compute_interactions(state, params,
                    neighbors,
                    nb_influent=2,
                    wall=None,
                    altitude=None, z_min = 4., z_max = 8.,
                    direction=None)
            noise = np.random.normal(0., NOISES)
            dt = 1. / self.control_freq_hz
            speed_cmd = state.get_speed_2d() + (cmd.a_bx + noise[1]) * dt
            vz_cmd = state.get_vz() + (cmd.a_z + noise[2]) * dt
            yaw_cmd = (cmd.a_by / max(state.get_speed_2d(),0.1)) + noise[0]
            speed = np.array([
                speed_cmd * math.cos(state.get_course(use_heading=True)),
                speed_cmd * math.sin(state.get_course(use_heading=True)),
                vz_cmd,
                yaw_cmd])
            commands[uav_id] = speed

        #### Log the simulation ####################################
        if i % LOG_EVERY_N_STEPS == 0:
            for j in range(ARGS.num_drones):
                logger.log(drone=j,
                           timestamp=i/ARGS.simulation_freq_hz,
                           state=sim[j].as_ndarray(),
                           control=commands[j]
                           )

    #### Save the simulation results ###########################
    if ARGS.save_quant:
        quant = compute_quantification(logger.get_positions(), logger.get_velocities(), int(logger.nb_steps * 0.2))
        file_name = os.path.join(ARGS.log_file_path, 'quant_'+ARGS.log_name+'.txt')
        #np.savez(file_name, quant=quant)
        np.savetxt(file_name, quant.reshape((1,6)))
    else:
        logger.save(file_path=ARGS.log_file_path, file_name=ARGS.log_name)

    return f'{ARGS.log_file_path}{ARGS.log_name} done'

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description="Phase diagram simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--num_drones", default=5, type=int, help="Number of drones", metavar="")
    parser.add_argument("--speed_setpoint", default=1., type=float, help="speed setpoint", metavar="")
    parser.add_argument("--simulation_freq_hz", default=20, type=int, help="Simulation frequency in Hz", metavar="")
    parser.add_argument("--log_freq_hz", default=10, type=int, help="Log frequency in Hz", metavar="")
    parser.add_argument("--log_name", default=None, type=str, help="Log file name", metavar="")
    parser.add_argument("--log_file_path", default='logs/', type=str, help="Log file path", metavar="")
    parser.add_argument("--duration_sec", default=20, type=int, help="Duration of the simulation in seconds", metavar="")
    parser.add_argument("--swarm_config", default='config/demo.yaml', type=str, help="SwarmFish parameter file", metavar="")
    parser.add_argument("--random_init", action='store_true', help="Randomize init (heading only)")
    parser.add_argument("--y_att", default=0.5, type=float, help="attraction coefficient", metavar="")
    parser.add_argument("--y_ali", default=0.5, type=float, help="aligment coefficient", metavar="")
    parser.add_argument("--save_quant", action='store_true', help="Save quantification instead of full log")
    ARGS = parser.parse_args()

    try:
        run_simulation(ARGS)
    except (KeyboardInterrupt, SystemExit):
        print("Simu phase diagram stopped")

# EOF
