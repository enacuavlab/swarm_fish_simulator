import sys

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QDockWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSlider
from PyQt6.QtCore import Qt, QTimer
from ui_swarm_controller import Ui_SwarmController

import pyqtgraph as pg
import pyqtgraph.opengl as gl
import utils.util_pyqtgraph as utqg
import trimesh
from scipy.spatial.transform import Rotation as R

import argparse
import dataclasses
import numpy as np
import math

from simple_sim.point_sim import PointSim, QuadrotorState

from swarmfish.utils import load_params_from_yaml
import swarmfish.swarm_control as sc

def ms_of_hz(freq):
    return int(1000 / freq)

SIMULATION_FREQ = 20 # in Hz
CONTROL_FREQ = 10  # in HZ
NB_OF_DRONES = 5


class SwarmFish_Environment():
    env = None

    def __init__(self, ARGS, create_env=True):
        self.num_drones = ARGS.num_drones
        self.simulation_freq_hz = float(ARGS.simulation_freq)
        self.time_step = 1. / self.simulation_freq_hz
        self.random_init = ARGS.random_init
        self.sim = []

        if create_env:
            # Initialize the simulation on a square
            dim = math.ceil(math.sqrt(self.num_drones))
            dist = 1.
            init_state = np.array([[ dist*(i % dim), dist*math.floor(i / dim), 0., 0. ] for i in range(self.num_drones)]) 
            if self.random_init:
                init_state[:,3] = 2.*np.pi*np.random.rand(self.num_drones)
            self.create(init_state)

    def create(self, init_state: np.ndarray):
        self.sim = [ PointSim(init_position=x0[0:3], init_yaw=x0[3]) for x0 in init_state ]

    def step(self, commands):
        for i in range(self.num_drones):
            self.sim[i].update_state(
                    desired_velocity=commands[i][0:3],
                    desired_yaw_rate=commands[i][3],
                    time_step=self.time_step)
        return self.get_states()

    def get_states(self):
        return [ self.sim[i].get_state() for i in range(self.num_drones) ]

    def close(self):
        pass


class SwamFish_View(QMainWindow):
    mesh_list = {}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Swarm Fish View")
        self.resize(1024,768)
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(100,100,100, 255)
        self.grid = gl.GLGridItem()
        self.grid.setColor('k')
        self.view.addItem(self.grid)
        self.setCentralWidget(self.view)
        self.show()

    def close(self):
        print("Closing view")
        QApplication.exit()

    # Create a cylinder in pybullet format
    def create_cylinder(self, radius:float, height:float, pos:np.ndarray, res:int=20):
        t_ = np.linspace(0, 2*np.pi, res)
        x_ = pos[0] + radius * np.sin(t_)
        y_ = pos[1] + radius * np.cos(t_)
        bottom = np.array([[x,y,pos[2]] for x,y in zip(x_,y_)])
        top = np.array([[x,y,pos[2]+height] for x,y in zip(x_,y_)])
        return np.vstack((bottom,top))
    
    def add_cylinder(self, radius:float, height:float, pos:np.ndarray, res:int=20, color=(1.,1.,1.,1.)):
        verts, faces = utqg.cylinder_mesh(radius, height, res)
        mesh = gl.GLMeshItem(vertexes=verts,faces=faces,drawFaces=True,
                    drawEdges=False,smooth=False,computeNormals=True,
                    shader='shaded',glOptions='opaque')
        mesh.setColor(color)
        mesh.translate(pos[0], pos[1], pos[2])
        self.view.addItem(mesh)

    def add_circle(self, radius:float, pos:np.ndarray, width:float=2., res:int=20, color=(1.,1.,1.,1.)):
        t = np.linspace(0,2*np.pi,res,endpoint=True).reshape(res,1)
        vertices = np.zeros((res,3))
        vertices[0:res,:] = np.hstack((radius*np.cos(t),radius*np.sin(t),np.zeros((res,1))))
        circle = gl.GLLinePlotItem(pos=vertices, color=color)
        circle.translate(pos[0], pos[1], pos[2])
        self.view.addItem(circle)

    def add_polygon(self, vertices:np.ndarray, height:float, color=(1., 1., 1., 1.)):
        verts, faces = utqg.polygon_mesh(vertices, height)
        mesh = gl.GLMeshItem(vertexes=verts,faces=faces,drawFaces=True,
                    drawEdges=False,smooth=False,computeNormals=True,
                    shader='shaded',glOptions='opaque')
        mesh.setColor(color)
        self.view.addItem(mesh)

    def add_object(self, mesh_id, mesh, text=None, color=(255, 0, 0, 255)):
        m = gl.GLMeshItem(vertexes=mesh.vertexes(), faces=mesh.faces(),
                    color=color, drawFaces=True,
                    drawEdges=False,smooth=False,computeNormals=False,
                    shader=None,glOptions='opaque')
        self.view.addItem(m)
        if text is not None:
            t = gl.GLTextItem(parentItem=m, pos=[0,0,0.2], text=text)
            self.view.addItem(t)
        self.mesh_list[mesh_id] = m
        self.move_object(mesh_id, np.array([0.,0.,0.]), np.array([0.,0.,0.]))
        self.view.update()

    def move_object(self, obj_id, pos, att):
        r = R.from_euler('xyz', att)
        axis = r.as_rotvec()
        angle = np.linalg.norm(axis)
        self.mesh_list[obj_id].resetTransform()
        self.mesh_list[obj_id].rotate(np.degrees(angle), *axis) # FIXME ? ,local=True)
        self.mesh_list[obj_id].translate(pos[0],pos[1],pos[2])

    def add_line(self, pos1, pos2, color=(255, 255, 255, 255)):
        line = gl.GLLinePlotItem(pos=np.vstack((pos1, pos2)), color=color)
        self.view.addItem(line)
        return line

    def move_line(self, line, pos1, pos2, color=None):
        if color is None:
            line.setData(pos=np.vstack((pos1, pos2)))
        else:
            line.setData(pos=np.vstack((pos1, pos2)), color=color)



class SwarmFish_Controller(QWidget, Ui_SwarmController):
    speed_setpoint = 1.
    direction_setpoint = None
    altitude_setpoint = 5.
    intruders_id = []

    def __init__(self, ARGS, env, view):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("SwarmFish")

        self.num_drones = ARGS.num_drones
        self.simulation_freq_hz = ARGS.simulation_freq
        self.control_freq_hz = ARGS.control_freq
        self.env = env
        self.view = view

        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.update_simulation)
        self.action_timer = QTimer()
        self.action_timer.timeout.connect(self.update_action)

        #### Initialize the controllers ############################
        self.current_time = 0.
        self.commands = [ np.zeros(4) for i in range(self.num_drones) ]
        self.obs = self.env.step(self.commands)

        for d in range(env.num_drones):
            mesh = trimesh.load_mesh(ARGS.mesh_file)
            mesh_data = gl.MeshData(vertexes=mesh.vertices, faces=mesh.faces)
            vertices = mesh_data.vertexes() / 1000.
            mesh_data.setVertexes(vertices)
            view.add_object(d, mesh_data, text=str(d))

        #### load params from config file 
        self.params = load_params_from_yaml(ARGS.swarm_config)

        self.make_layout()
        self.show()

    def update_simulation(self):
        self.current_time += 1. / self.simulation_freq_hz
        self.obs = self.env.step(self.commands)
        for d in range(self.env.num_drones):
            self.view.move_object(d, self.obs[d].pos, self.obs[d].att)

    def update_action(self):
        raise NotImplementedError

    def start_simulation(self):
        self.action_timer.start(ms_of_hz(self.control_freq_hz))
        self.simulation_timer.start(ms_of_hz(self.simulation_freq_hz))
        print("simultion started")

    def stop_simulation(self):
        self.simulation_timer.stop()
        self.action_timer.stop()
        print("simultion stoped")

    def close(self):
        print("Closing controller")
        self.stop_simulation()

    def make_layout(self):
        CURSOR_RES = 1000

        # Boutons
        self.btn_start.clicked.connect(lambda: self.start_simulation())
        self.btn_stop.clicked.connect(lambda: self.stop_simulation())
        self.btn_quit.clicked.connect(lambda: self.view.close())

        # Speed
        def update_speed(_v):
            self.speed_setpoint = _v / 100.
            self.label_speed.setText(str(_v / 100.))
        self.slider_speed.valueChanged.connect(update_speed)

        # Direction
        def update_direction(_v):
            if self.check_dir.isChecked():
                self.direction_setpoint = np.radians(_v)
                self.label_dir.setText(str(_v))
            else:
                self.direction_setpoint = None
        self.slider_dir.valueChanged.connect(update_direction)
        self.check_dir.stateChanged.connect(lambda _: update_direction(self.slider_dir.value()))

        # Altitude
        def update_alt(_v):
            if self.check_alt.isChecked():
                self.altitude_setpoint = _v / 10.
                self.label_alt.setText(str(_v / 10.))
            else:
                self.altitude_setpoint = None
        self.slider_alt.valueChanged.connect(update_alt)
        self.check_alt.stateChanged.connect(lambda _: update_alt(self.slider_alt.value()))

        # Intruders
        def update_intruders():
            try:
                self.intruders_id = [int(i) for i in self.intruders_input.text().split(',')]
                print(f'Set intruders ID: {self.intruders_id}')
            except:
                self.intruders_id = []
                print('Invalid or no intruders ID')
        self.intruders_input.returnPressed.connect(update_intruders)

        try:
            # Parameters
            p_dict = dataclasses.asdict(self.params)
            exclude = ['max_velocity','min_velocity','zmax','zmin','use_heading']
            parameters = [ p for p in p_dict.keys() if p not in exclude ]

            container_widget = QWidget()
            param_layout = QVBoxLayout()

            for label_text in parameters:
                label = QLabel(label_text)
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setMinimum(0)
                slider.setMaximum(int(max(p_dict[label_text] * 5 * CURSOR_RES,5)))
                slider.setValue(int(p_dict[label_text] * CURSOR_RES))

                # Label pour afficher la valeur courante du slider
                slider_label = QLabel(str(p_dict[label_text]))
                horiz = QHBoxLayout()
                horiz.addWidget(label)
                horiz.addWidget(slider)
                horiz.addWidget(slider_label)

                def dataclass_from_dict(da, d):
                    try:
                        fieldtypes = {f.name:f.type for f in dataclasses.fields(da)}
                        return da(**{f:dataclass_from_dict(fieldtypes[f],d[f]) for f in d})
                    except:
                        return d # Not a dataclass field

                # Mettre à jour le label lorsque la valeur du slider change
                def update_val(value,l=label_text,s=slider_label):
                    var = float(value / CURSOR_RES)
                    tmp = dataclasses.asdict(self.params)
                    tmp[l] = var
                    self.params = dataclass_from_dict(sc.SwarmParams, tmp)
                    s.setText(str(value / CURSOR_RES))
                slider.valueChanged.connect(update_val)
                param_layout.addLayout(horiz)

            container_widget.setLayout(param_layout)
            self.params_area.setWidget(container_widget)
        except Exception as e:
            print("Fail to load parameters",e)
            sys.exit(0)

        self.show()


def make_args_parser():
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description="SwarmFish control with Qt")
    parser.add_argument("--num_drones", default=NB_OF_DRONES, type=int, help="Number of drones")
    parser.add_argument("--simulation_freq", default=SIMULATION_FREQ, type=int, help="Simulation frequency in Hz")
    parser.add_argument("--control_freq", default=CONTROL_FREQ, type=int, help="Control frequency in Hz")
    parser.add_argument("--swarm_config", default='config/demo.yaml', type=str, help="SwarmFish parameter file")
    parser.add_argument("--mesh_file", default='models/triangle.stl', type=str, help="STL drone file")
    parser.add_argument("--random_init", action='store_true', help="Randomize init (heading only)")
    return parser


if __name__ == "__main__":
    pass

