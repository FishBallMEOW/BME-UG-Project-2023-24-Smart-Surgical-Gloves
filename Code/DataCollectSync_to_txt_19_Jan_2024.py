from serial import Serial
import math
from sksurgerynditracker.nditracker import NDITracker
from threading import Thread
import time
import ctypes
import os
import pyglet 
from pyglet.gl import *
from pyglet.window import mouse
from pywavefront import visualization, Wavefront
import csv
import numpy as np
from PyQt6 import QtWidgets, QtCore
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication


#---------------------------------Import---------------------------------------------------------------------------------------------------------

root_path = os.path.dirname(__file__)
# hand obj
hand_obj = Wavefront(os.path.join(root_path, 'Object/hand/right_hand.obj'))
hand_red_obj = Wavefront(os.path.join(root_path, 'Object/hand/hand_red.obj'))
hand_orange_obj = Wavefront(os.path.join(root_path, 'Object/hand/hand_orange.obj'))
hand_yellow_obj = Wavefront(os.path.join(root_path, 'Object/hand/hand_yellow.obj'))
hand_green_obj = Wavefront(os.path.join(root_path, 'Object/hand/hand_green.obj'))
# prostate obj
prostate_obj = Wavefront(os.path.join(root_path, 'Object/prostate/prostate_realistic.obj'))
prostate_red_RT_obj = Wavefront(os.path.join(root_path, 'Object/prostate/prostate_realistic_cut_RT_red.obj'))
prostate_red_RB_obj = Wavefront(os.path.join(root_path, 'Object/prostate/prostate_realistic_cut_RB_red.obj'))
prostate_red_LT_obj = Wavefront(os.path.join(root_path, 'Object/prostate/prostate_realistic_cut_LT_red.obj'))
prostate_red_LB_obj = Wavefront(os.path.join(root_path, 'Object/prostate/prostate_realistic_cut_LB_red.obj'))
# plane obj
plane_obj = Wavefront(os.path.join(root_path, 'Object/hand/plane.obj'))

#---------------------------------Setup---------------------------------------------------------------------------------------------------------
# set up serial connection to Arduino
sPort_Arduino = Serial('COM4', 9600, timeout=0.5)  # replace with the correct port and baud rate

#  set up the NDITracker
settings_aurora = { "tracker type": "aurora",
                   "ports to probe": 2,
                    "verbose": True,}
tracker = NDITracker(settings_aurora)
tracker.use_quaternions = "true"

#---------------------------------Variables---------------------------------------------------------------------------------------------------------
viewport_width=1280
viewport_height=720
rotation = 0.0
timer = 0.0
lightfv = ctypes.c_float * 4
x_i = 0.0
y_i = 0.0
z_i = 0.0
x1 = 0.0
y1 = 0.0
z1 = -30.0
rot_x1 = 0.0
rot_y1 = 0.0
rot_z1 = 0.0
x2 = 0.0
y2 = 0.0
z2 = -30.0
rot_x2 = 0.0
rot_y2 = 0.0
rot_z2 = 0.0
steps = -1
trigger = False
stress = []
strain = []
x_offset = 0
y_offset = -3
z_offset = -8 
csvfileName = 'data_20_feb_2024\data_0010_same_pos1.csv'  # change it to the desired file (.csv) location
zoom = 2
rot_cam = (0, 0)
cam_pos = (0, 0, 0)
#---------------------------------Functions&Classes-----------------------------------------------------------------------------------------------
# Setting the Thread function with returns
class ThreadWithReturnValue(Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

# window for the plots
class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        self.x = list(range(100))  # 100 time points
        self.y = [0 for _ in range(100)]  # 100 data points

        self.graphWidget.setBackground('w')

        pen = pg.mkPen(color=(255, 0, 0), width=3)
        self.data_line =  self.graphWidget.plot(self.x, self.y, pen=pen)
    
    def set_title(self, title):
        self.graphWidget.setTitle(title, color="b", size="30pt")

    def update_plot_data(self, data):

        self.x = self.x[1:]  # Remove the first y element.
        self.x.append(self.x[-1] + 1)  # Add a new value 1 higher than the last.

        self.y = self.y[1:]  # Remove the first
        self.y.append(data)  # Add a new recent value.

        self.data_line.setData(self.x, self.y)  # Update the data.

def readLocation():
    # Read the data from the Location sensor
    dataLocation = [] 
    for i1 in range(1):
        port_handles, timestamps, framenumbers, tracking, quality = tracker.get_frame()
        dataLocation.append([tracking, timestamps])
    return dataLocation

def readPressure():
    # Read the data from the Pressure sensor 
    dataPressure = []
    for i1 in range(1):
        dataPoint = sPort_Arduino.readline().decode().strip() 
        if dataPoint == '':
            dataPoint = '0' 
        timestamps = time.time()
        dataPressure.append([float(dataPoint), float(timestamps)])
    return dataPressure

def q2e(qw, qx, qy, qz):
    ysqr = qy * qy

    t0 = +2.0 * (qw * qx + qy * qz)
    t1 = +1.0 - 2.0 * (qx * qx + ysqr)
    X = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (qw * qy - qz * qx)

    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    Y = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (qw * qz + qx * qy)
    t4 = +1.0 - 2.0 * (ysqr + qz * qz)
    Z = np.degrees(np.arctan2(t3, t4))

    return X, Y, Z


def aurora2opengl(x,y,z):
    x = -x/300*2
    y = (-y/300-1)*2*float(viewport_height)/viewport_width
    z = (z-210)/420*10
    return [x,y,z]

def draw_object(obj, x, y, z, rot_x, rot_y, rot_z):
    global rot_cam, cam_pos

    glLoadIdentity()
    glPushMatrix

    rot_cam_x, rot_cam_y = rot_cam
    cam_x, cam_y, cam_z = cam_pos
    print("1", rot_cam)
    glRotatef(rot_cam_x, 0, 1, 0)
    glRotatef(-rot_cam_y, math.cos(math.radians(rot_cam_x)), 0, math.sin(math.radians(rot_cam_x)))
    glTranslated(cam_x, cam_y, cam_z)

    glTranslated(x, y, z)
    glRotatef(rot_x, -1.0, 0.0, 0.0)
    glRotatef(rot_y, 0.0, -1.0, 0.0)
    glRotatef(rot_z, 0.0, 0.0, 1.0)
    visualization.draw(obj)
    glPopMatrix


def update(dt):
    global x1, y1, z1, rot_x1, rot_y1, rot_z1, x2, y2, z2, rot_x2, rot_y2, rot_z2, pressure, trigger, x_i, y_i, z_i, stress, strain
    t_readLocation = ThreadWithReturnValue(target=readLocation, name='t_readLocation')
    t_readPressure = ThreadWithReturnValue(target=readPressure, name='t_readPressure')

    t_readLocation.start()
    t_readPressure.start()

    dataLocation = t_readLocation.join()
    dataPressure = t_readPressure.join()

    pressure = dataPressure[0][0]

    [z1, x1, y1] = dataLocation[0][0][0][0][4:7]
    [z2, x2, y2] = dataLocation[0][0][1][0][4:7]
    print('BEFORE:', 'x2:', x2, 'y2:', y2, 'z2:', z2)
    [x1, y1, z1] = aurora2opengl(x1, y1, z1)
    [x2, y2, z2] = aurora2opengl(x2, y2, z2)

    rot_z1, rot_x1, rot_y1  = q2e(dataLocation[0][0][0][0][0],dataLocation[0][0][0][0][1],dataLocation[0][0][0][0][2],dataLocation[0][0][0][0][3])
    rot_z2, rot_x2, rot_y2  = q2e(dataLocation[0][0][1][0][0],dataLocation[0][0][1][0][1],dataLocation[0][0][1][0][2],dataLocation[0][0][1][0][3])  # transformed the coordinate system (aurora: x,y,z --> opengl: z, x, y)

    #print('1', dataLocation[0][0][0][0][4], dataLocation[0][0][1][0][4], dataPressure[0][0])
    print('x2:', x2, 'y2:', y2, 'z2:', z2)
    with open(csvfileName, 'a', newline='') as csvfile:
        fieldnames = ['timestamps', 'qw1', 'qx1', 'qy1', 'qz1', 'x1', 'y1', 'z1', 'qw2', 'qx2', 'qy2', 'qz2', 'x2', 'y2', 'z2', 'pressure']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'timestamps': dataLocation[0][1][0], 'qw1':dataLocation[0][0][0][0][0], 'qx1':dataLocation[0][0][0][0][1], 'qy1':dataLocation[0][0][0][0][2], 
                        'qz1':dataLocation[0][0][0][0][3], 'x1':dataLocation[0][0][0][0][4], 'y1':dataLocation[0][0][0][0][5], 'z1':dataLocation[0][0][0][0][6]})
        writer.writerow({'timestamps': dataLocation[0][1][1], 'qw2':dataLocation[0][0][1][0][0], 'qx2':dataLocation[0][0][1][0][1], 'qy2':dataLocation[0][0][1][0][2], 
                        'qz2':dataLocation[0][0][1][0][3], 'x2':dataLocation[0][0][1][0][4], 'y2':dataLocation[0][0][1][0][5], 'z2':dataLocation[0][0][1][0][6]})
        writer.writerow({'timestamps':dataPressure[0][1], 'pressure':dataPressure[0][0]})
    csvfile.close

    # update for the plot
    force_w.update_plot_data(pressure)


#---------------------------------Initialization------------------------------------------------------------------------------------------------
#Initialize aurora
tracker.start_tracking()

# Creating a window
window = pyglet.window.Window(viewport_width, viewport_height, "Surgical Gloves", resizable=True)

# initialize csvwriter
with open(csvfileName, 'w', newline='') as csvfile:
    fieldnames = ['timestamps', 'qw1', 'qx1', 'qy1', 'qz1', 'x1', 'y1', 'z1', 'qw2', 'qx2', 'qy2', 'qz2', 'x2', 'y2', 'z2', 'pressure']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
csvfile.close

# Initialize the widget for the plot of pressure
app = QtWidgets.QApplication(sys.argv)
force_w = MainWindow()
force_w.set_title("Pressure")
force_w.show()

#---------------------------------Loop----------------------------------------------------------------------------------------------------------
# on the event of creating/ drawing the window
@window.event  
def on_draw():  
    window.clear()
    glLoadIdentity()
    glLightfv(GL_LIGHT0, GL_POSITION, lightfv(-1.0, 1.0, 1.0, 0.0))
    draw_object(plane_obj, 0, -2, -5, 0, 0, 0)    #draw_object(prostate_obj, x1, y1, z1, 0, 0, 0)
    draw_object(hand_obj, x2, y2, z2, rot_x2, rot_y2, rot_z2)
    #draw_object()

    if pressure >= 0.5:
        print(x2-(x1-25/300*2), -z2+z1)
        if x2-(x1-25/300*2)>=0 and -z2+z1>=0:
            draw_object(prostate_red_RT_obj, x1, y1, z1, 0, 0, 0)
        elif x2-(x1-25/300*2)>=0 and -z2+z1<=0:
            draw_object(prostate_red_RB_obj, x1, y1, z1, 0, 0, 0)
        elif x2-(x1-25/300*2)<0 and -z2+z1>0:
            draw_object(prostate_red_LT_obj, x1, y1, z1, 0, 0, 0)
        else: 
            draw_object(prostate_red_LB_obj, x1, y1, z1, 0, 0, 0)
    else:
        draw_object(prostate_obj, x1, y1, z1, 0, 0, 0)

# on the event of resizing the window
@window.event
def on_resize(width, height):  
    global viewport_height, viewport_width
    viewport_width, viewport_height = window.get_framebuffer_size()
    glViewport(0, 0, viewport_width, viewport_height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glFrustum(-2, 2, -2*float(viewport_height)/viewport_width, 2*float(viewport_height)/viewport_width, 1., 10.)
    glMatrixMode(GL_MODELVIEW)
    return True

@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    global zoom, viewport_height, viewport_width
    if not(zoom <=1 and scroll_y <0):
        zoom += scroll_y/5
    print(zoom)
    glViewport(0, 0, viewport_width, viewport_height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glFrustum(-zoom, zoom, -zoom*float(viewport_height)/viewport_width, zoom*float(viewport_height)/viewport_width, 1., 100.)
    glMatrixMode(GL_MODELVIEW)
    pass

@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    global rot_cam
    m = 0.15
    x, y = rot_cam
    x, y = x + dx * m, y + dy * m
    y = max(-90, min(90, y))
    rot_cam = (x,y)
    print(rot_cam)
    pass

@window.event
def on_key_press(symbol, modifiers):
    global cam_pos
    x, y, z= cam_pos
    if symbol == key.W:
        z -= 1                
    elif symbol == key.S:
        z += 1
    elif symbol == key.A:
        x += 1
    elif symbol == key.D:
        x -= 1
    elif symbol == key.SPACE or symbol == key.UP:
        y -= 1
    elif symbol == key.DOWN:
        y += 1
    cam_pos = (x,y,z)
    pass

pyglet.clock.schedule(update)
pyglet.app.run()


#-----------------------------------------------------------------------------------------------------------------------------------------------


