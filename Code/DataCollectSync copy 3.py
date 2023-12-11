import serial
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


#---------------------------------Setup---------------------------------------------------------------------------------------------------------
# set up serial connection to Arduino
sPort_Arduino = serial.Serial('COM5', 9600, timeout=0.5)  # replace with the correct port and baud rate

#  set up the NDITracker
settings_aurora = { "tracker type": "aurora",
                   "ports to probe": 2,
                    "verbose": True,}
tracker = NDITracker(settings_aurora)
tracker.use_quaternions = "true"

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

def readLocation():
    # Read the data from the Location sensor
    dataLocation = [] 
    for i1 in range(20):
        port_handles, timestamps, framenumbers, tracking, quality = tracker.get_frame()
        dataLocation.append([tracking, timestamps])
    return dataLocation

def readPressure():
    # Read the data from the Pressure sensor 
    dataPressure = []
    for i1 in range(20):
        dataPoint = sPort_Arduino.readline().decode().strip() 
        if dataPoint == '':
            dataPoint = '0' 
        timestamps = time.time()
        dataPressure.append([int(float(dataPoint)), int(float(timestamps))])
    return dataPressure

# Creating a window
window = pyglet.window.Window(viewport_width, viewport_height, "Surgical Gloves", resizable=True)

# on the event of creating/ drawing the window
@window.event  
def on_draw():  
    window.clear()
    glLoadIdentity()
    glLightfv(GL_LIGHT0, GL_POSITION, lightfv(-1.0, 1.0, 1.0, 0.0))
    draw_object(hand_obj, x1, y1, z1, rot_x1, rot_y1, rot_z1)

# on the event of resizing the window
@window.event
def on_resize(width, height):  
    global viewport_height, viewport_width
    viewport_width, viewport_height = window.get_framebuffer_size()
    glViewport(0, 0, viewport_width, viewport_height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glFrustum(-1, 1, -float(viewport_height)/viewport_width, float(viewport_height)/viewport_width, 1., 100.)
    glMatrixMode(GL_MODELVIEW)
    return True

def draw_object(obj, x, y, z, rot_x, rot_y, rot_z):
    glLoadIdentity()
    glTranslated(x, y, z)
    glRotatef(rot_y, 0.0, 1.0, 0.0)
    glRotatef(rot_x, 1.0, 0.0, 0.0)
    glRotatef(rot_z, 0.0, 0.0, 1.0)
    visualization.draw(obj)


def update(dt):
    global x1, y1, z1, rot_x1, rot_y1, rot_z1
    t_readLocation = ThreadWithReturnValue(target=readLocation, name='t_readLocation')
    t_readPressure = ThreadWithReturnValue(target=readPressure, name='t_readPressure')

    t_readLocation.start()
    t_readPressure.start()

    dataLocation = t_readLocation.join()
    dataPressure = t_readPressure.join()

    [qw, rot_x1, rot_y1, rot_z1, x1, y1, z1] = dataLocation[0][1]

#---------------------------------Initialization------------------------------------------------------------------------------------------------
#Initialize variables
root_path = os.path.dirname(__file__)
hand_obj = Wavefront(os.path.join(root_path, 'hand/hand.obj'))
viewport_width=1280
viewport_height=720
rotation = 0.0
timer = 0.0
lightfv = ctypes.c_float * 4
x1 = 0.0
y1 = 0.0
z1 = 0.0
steps = -1

#Initialize aurora
tracker.start_tracking()
window = pyglet.window.Window(viewport_width, viewport_height, "Surgical Gloves", resizable=True)

#---------------------------------Loop----------------------------------------------------------------------------------------------------------
pyglet.clock.schedule(update)
pyglet.app.run()


#-----------------------------------------------------------------------------------------------------------------------------------------------


