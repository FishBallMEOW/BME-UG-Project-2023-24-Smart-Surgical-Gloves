import math
from sksurgerynditracker.nditracker import NDITracker
from threading import Thread
import time
import ctypes
import os
import pyglet 
from pyglet.gl import *
from pywavefront import visualization, Wavefront
import pandas as pd
import csv
import numpy as np
from PyQt6 import QtWidgets, QtCore
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
from pyglet.window import key, mouse
import numpy.polynomial.polynomial as poly
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn import metrics
from Utils import RealTimePlotter
from Utils import ControlBox
#---------------------------------Import---------------------------------------------------------------------------------------------------------

root_path = os.path.dirname(__file__)
data_file = 'data/20_feb_2024/data_20_feb_2024/data_0050_same_pos.csv'  # change to desire data (.csv) file name
stress_strain_csv = os.path.join(root_path, data_file[:-4]+'_stress_strain.csv')
# hand obj
hand_obj = Wavefront(os.path.join(root_path, 'Object/hand/right_hand.obj'))
# hand_red_obj = Wavefront(os.path.join(root_path, 'Object/hand/hand_red.obj'))
# hand_orange_obj = Wavefront(os.path.join(root_path, 'Object/hand/hand_orange.obj'))
# hand_yellow_obj = Wavefront(os.path.join(root_path, 'Object/hand/hand_yellow.obj'))
# hand_green_obj = Wavefront(os.path.join(root_path, 'Object/hand/hand_green.obj'))

# prostate obj
prostate_obj = Wavefront(os.path.join(root_path, 'Object/prostate/prostate_realistic.obj'))
# prostate_red_RT_obj = Wavefront(os.path.join(root_path, 'Object/prostate/prostate_realistic_cut_RT_red.obj'))
# prostate_red_RB_obj = Wavefront(os.path.join(root_path, 'Object/prostate/prostate_realistic_cut_RB_red.obj'))
# prostate_red_LT_obj = Wavefront(os.path.join(root_path, 'Object/prostate/prostate_realistic_cut_LT_red.obj'))
# prostate_red_LB_obj = Wavefront(os.path.join(root_path, 'Object/prostate/prostate_realistic_cut_LB_red.obj'))

# prostate obj more cut 
prostate_red_1_obj = Wavefront(os.path.join(root_path, 'Object/prostate/More Cut/prostate_realistic_more_cut_1.obj'))
prostate_red_2_obj = Wavefront(os.path.join(root_path, 'Object/prostate/More Cut/prostate_realistic_more_cut_2.obj'))
prostate_red_3_obj = Wavefront(os.path.join(root_path, 'Object/prostate/More Cut/prostate_realistic_more_cut_3.obj'))
prostate_red_4_obj = Wavefront(os.path.join(root_path, 'Object/prostate/More Cut/prostate_realistic_more_cut_4.obj'))
prostate_red_5_obj = Wavefront(os.path.join(root_path, 'Object/prostate/More Cut/prostate_realistic_more_cut_5.obj'))
prostate_red_6_obj = Wavefront(os.path.join(root_path, 'Object/prostate/More Cut/prostate_realistic_more_cut_6.obj'))
prostate_red_7_obj = Wavefront(os.path.join(root_path, 'Object/prostate/More Cut/prostate_realistic_more_cut_7.obj'))
prostate_red_8_obj = Wavefront(os.path.join(root_path, 'Object/prostate/More Cut/prostate_realistic_more_cut_8.obj'))
prostate_red_9_obj = Wavefront(os.path.join(root_path, 'Object/prostate/More Cut/prostate_realistic_more_cut_9.obj'))
prostate_red_10_obj = Wavefront(os.path.join(root_path, 'Object/prostate/More Cut/prostate_realistic_more_cut_10.obj'))
prostate_red_11_obj = Wavefront(os.path.join(root_path, 'Object/prostate/More Cut/prostate_realistic_more_cut_11.obj'))
prostate_red_12_obj = Wavefront(os.path.join(root_path, 'Object/prostate/More Cut/prostate_realistic_more_cut_12.obj'))

# plane obj
plane_obj = Wavefront(os.path.join(root_path, 'Object/hand/plane.obj'))
# dataframe
df_data = pd.read_csv(os.path.join(root_path, data_file))

#---------------------------------Data Preprocess---------------------------------------------------------------------------------------------------------
df_location_1 = df_data.copy()[['timestamps', 'qw1', 'qx1', 'qy1', 'qz1', 'x1', 'y1', 'z1']]
df_location_2 = df_data.copy()[['timestamps', 'qw2', 'qx2', 'qy2', 'qz2', 'x2', 'y2', 'z2']]
df_pressure = df_data.copy()[['timestamps', 'pressure']]

# separate the data of two location sensor and pressure
i1 = list(range(0, len(df_data.index), 3))
i2 = list(range(1, len(df_data.index), 3))
i3 = list(range(2, len(df_data.index), 3))
df_location_1 = df_location_1.loc[i1]
df_location_2 = df_location_2.loc[i2]
df_pressure = df_pressure.loc[i3]

#---------------------------------Variables---------------------------------------------------------------------------------------------------------
viewport_width=1280
viewport_height=1080
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
t = 0
trigger = False
stress = []
strain = 0.0
strain_diff = 0.0
moving_ave_temp = []
time_pressure = []
zoom = 2
rot_cam = (0, 0)
cam_pos = (0, 0, 0)
close_bool = False

#---------------------------------Functions&Classes-----------------------------------------------------------------------------------------------
class ThreadWithReturnValue(Thread):
    # Setting the Thread function with returns
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

def readLocation(t):
    # Read the data from the Location sensor
    dataLocation = [] 
    tracking_1 = df_location_1.iloc[t, 1:8].tolist()
    tracking_2 = df_location_2.iloc[t, 1:8].tolist()
    timestamps_1 = df_location_1.iloc[t, 0].tolist()
    timestamps_2 = df_location_2.iloc[t, 0].tolist()
    dataLocation.append([tracking_1, timestamps_1])
    dataLocation.append([tracking_2, timestamps_2])
    return dataLocation

def readPressure(t):
    # Read the data from the Pressure sensor 
    dataPressure = [] 
    pressure = df_pressure.iloc[t, 1].tolist()
    timestamps = df_pressure.iloc[t, 0].tolist()
    dataPressure.append([pressure, timestamps])
    return dataPressure

def q2e(qw, qx, qy, qz):
    # translate quaternion to euler angle
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
    """
    translate the input coordinates from aurora to opengl coordinates
    """
    x = -x/300*2
    y = (-y/300-1)*2*float(viewport_height)/viewport_width
    z = (z-210)/420*10
    return [x,y,z]

def distance_ori(x1, y1, z1, x2, y2, z2):
    # calculate distance after force is applied
    return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

def draw_object(obj, x, y, z, rot_x, rot_y, rot_z):
    global rot_cam, cam_pos

    glLoadIdentity()
    glPushMatrix()

    rot_cam_x, rot_cam_y = rot_cam
    cam_x, cam_y, cam_z = cam_pos
    glRotatef(rot_cam_x, 0, 1, 0)
    glRotatef(-rot_cam_y, math.cos(math.radians(rot_cam_x)), 0, math.sin(math.radians(rot_cam_x)))
    glTranslated(cam_x, cam_y, cam_z)

    glTranslated(x, y, z)
    glRotatef(rot_x, -1.0, 0.0, 0.0)  # negative because of the conversion of the coordinates
    glRotatef(rot_y, 0.0, -1.0, 0.0)  # same as x
    glRotatef(rot_z, 0.0, 0.0, 1.0)
    visualization.draw(obj)
    glPopMatrix()

def update(dt):
    global close_bool, x1, y1, z1, rot_x1, rot_y1, rot_z1, x2, y2, z2, rot_x2, rot_y2, rot_z2, t, pressure, trigger, x_i, y_i, z_i, stress, strain, strain_diff, moving_ave_temp

    if t == len(df_location_1.index):
        t = 0  # reset t to 0 to loop the data again
    
    t_readLocation = ThreadWithReturnValue(target=readLocation, name='t_readLocation', args=(t,))
    t_readPressure = ThreadWithReturnValue(target=readPressure, name='t_readPressure', args=(t,))

    t_readLocation.start()
    t_readPressure.start()

    dataLocation = t_readLocation.join()
    dataPressure = t_readPressure.join()

    t += 1
    area = math.pi* (9.53*(10**-3)/2)**2  # area of the flexiforce sensor
    pressure = (dataPressure[0][0]/area)/1000

    [qw1, rot_z1, rot_x1, rot_y1, z1, x1, y1] = dataLocation[0][0]  # aurora (x,y,z) --> opengl (z, x, y)
    [qw2, rot_z2, rot_x2, rot_y2, z2, x2, y2] = dataLocation[1][0]

    rot_x1, rot_y1, rot_z1 = q2e(qw1, rot_x1, rot_y1, rot_z1)
    rot_x2, rot_y2, rot_z2 = q2e(qw2, rot_x2, rot_y2, rot_z2)
    
    # moving average 
    if len(moving_ave_temp) <= 10:
        moving_ave_temp.append(pressure)
    else:
        moving_ave_temp = moving_ave_temp[1:]
        moving_ave_temp.append(pressure)

    pressure_moving_ave = float(sum(moving_ave_temp))/max(len(moving_ave_temp), 1)
    Pressure_w.update_plot_data(pressure_moving_ave)  #pressure)

    pressure_diff = Pressure_w.pressure_diff()

    # start the sampling for the stress-strain plot
    if pressure_moving_ave >= 2 and pressure_moving_ave <= 30 and not trigger:  # threshold: 2; remove abnormal pressure > 30
        x_i, y_i, z_i = x2, y2, z2
        trigger = True

    if pressure_diff <= 0 and trigger:  # limit weird data point with pressure_diff > 5
        x_i, y_i, z_i = 0.0, 0.0, 0.0
        Stress_strain_w.regression_each_press(False, False, False) 
        Stiff_LR = Stress_strain_w.regression_each_press(True)
        Stress_strain_w.set_title(f"Stiffness(each press): {round(Stiff_LR, 2)}")
        trigger = False

    # updating data if trigger is started and not yet stopped
    L_0 = 25  # formula to calculate strain = deformation (L)/ original length (L_0), Here for the phantom, it is assumed to be the radius (25mm)
    strain = (distance_ori(x_i, y_i, z_i, x2, y2, z2)/L_0)  # L_0 in mm and distance_ori(x_i, y_i, z_i, x2, y2, z2) also in mm
    if trigger: 
        Stress_strain_w.update_plot_data(strain, pressure_moving_ave)
        # write stress and strain to csv
        with open(stress_strain_csv, 'a', newline='') as csvfile:
            fieldnames = ['Stress', 'Strain']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'Stress':pressure_moving_ave, 'Strain':strain})
        csvfile.close

    # aurora to opengl coordinates
    [x1, y1, z1] = aurora2opengl(x1, y1, z1)
    [x2, y2, z2] = aurora2opengl(x2, y2, z2)

    time.sleep(0.02)

    close_bool = controlBox.return_close_bool()
    if close_bool:
        window.close()
        Pressure_w.close()
        Stress_strain_w.close()

#---------------------------------Initialization------------------------------------------------------------------------------------------------
# Creating a window
window = pyglet.window.Window(viewport_width, viewport_height, "3D Graphics Simulation", resizable=True)
window.set_minimum_size(600, 500)
window.set_location(0, 35)

# initialize csvwriter
with open(stress_strain_csv, 'w', newline='') as csvfile:
    fieldnames = ['Stress', 'Strain']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
csvfile.close

# Initialize the widget for the plot of pressure
app = QtWidgets.QApplication(sys.argv)
Pressure_w = RealTimePlotter.MainWindow()
Pressure_w.set_title("Pressure")
Pressure_w.show()
Stress_strain_w = RealTimePlotter.MainWindow_wo_x_lim()
Stress_strain_w.set_title("Stress-Strain Graph")
Stress_strain_w.show()

# Pop-up ControlBox
controlBox = ControlBox.MainWindow()


#---------------------------------Loop----------------------------------------------------------------------------------------------------------
@window.event  # on the event of creating/ drawing the window
def on_draw():  
    window.clear()
    glLoadIdentity()
    glLightfv(GL_LIGHT0, GL_POSITION, lightfv(-1.0, 1.0, 1.0, 0.0))
    draw_object(plane_obj, 0, -2, -5, 0, 0, 0)
    # draw_object(prostate_red_7_obj, 0, 0, -5, 0, 0, 0)
    draw_object(hand_obj, x2, y2, z2, rot_x2, rot_y2, rot_z2)

    # Split more prostate
    if pressure >= 0.5:
        # print(x2-(x1-25/300*2), -z2+z1)
        if -z2+z1>0.5 and -z2+z1<=1.5:
            if x2-(x1-25/300*2)>=-2 and x2-(x1-25/300*2)<=-1:
                draw_object(prostate_red_1_obj, x1, y1, z1, 0, 0, 0)
                # print('1')
            elif x2-(x1-25/300*2)>-1 and x2-(x1-25/300*2)<=0:
                draw_object(prostate_red_2_obj, x1, y1, z1, 0, 0, 0)
                # print('2')
            elif x2-(x1-25/300*2)>0 and x2-(x1-25/300*2)<=1:
                draw_object(prostate_red_3_obj, x1, y1, z1, 0, 0, 0)
                # print('3')
            elif x2-(x1-25/300*2)>0 and x2-(x1-25/300*2)<=2:
                draw_object(prostate_red_4_obj, x1, y1, z1, 0, 0, 0)
                # print('4')
        elif -z2+z1>-0.5 and -z2+z1<=0.5:
            if x2-(x1-25/300*2)>=-2 and x2-(x1-25/300*2)<=-1:
                draw_object(prostate_red_5_obj, x1, y1, z1, 0, 0, 0)
                # print('5')
            elif x2-(x1-25/300*2)>-1 and x2-(x1-25/300*2)<=0:
                draw_object(prostate_red_6_obj, x1, y1, z1, 0, 0, 0)
                # print('6')
            elif x2-(x1-25/300*2)>0 and x2-(x1-25/300*2)<=1:
                draw_object(prostate_red_7_obj, x1, y1, z1, 0, 0, 0)
                # print('7')
            elif x2-(x1-25/300*2)>0 and x2-(x1-25/300*2)<=2:
                draw_object(prostate_red_8_obj, x1, y1, z1, 0, 0, 0)
                # print('8')
        elif -z2+z1>-1.5 and -z2+z1<=-0.5:
            if x2-(x1-25/300*2)>=-2 and x2-(x1-25/300*2)<=-1:
                draw_object(prostate_red_9_obj, x1, y1, z1, 0, 0, 0)
                # print('9')
            elif x2-(x1-25/300*2)>-1 and x2-(x1-25/300*2)<=0:
                draw_object(prostate_red_10_obj, x1, y1, z1, 0, 0, 0)
                # print('10')
            elif x2-(x1-25/300*2)>0 and x2-(x1-25/300*2)<=1:
                draw_object(prostate_red_11_obj, x1, y1, z1, 0, 0, 0)
                # print('11')
            elif x2-(x1-25/300*2)>0 and x2-(x1-25/300*2)<=2:
                draw_object(prostate_red_12_obj, x1, y1, z1, 0, 0, 0)
                # print('12')
        else:
            draw_object(prostate_obj, x1, y1, z1, 0, 0, 0)
    else:
        draw_object(prostate_obj, x1, y1, z1, 0, 0, 0)


    # Split prostate
    # if pressure >= 2.0:
    #     if x2-(x1-25/300*2)>=0 and -z2+z1>=0:
    #         draw_object(prostate_red_RT_obj, x1, y1, z1, 0, 0, 0)
    #     elif x2-(x1-25/300*2)>=0 and -z2+z1<=0:
    #         draw_object(prostate_red_RB_obj, x1, y1, z1, 0, 0, 0)
    #     elif x2-(x1-25/300*2)<0 and -z2+z1>0:
    #         draw_object(prostate_red_LT_obj, x1, y1, z1, 0, 0, 0)
    #     else: 
    #         draw_object(prostate_red_LB_obj, x1, y1, z1, 0, 0, 0)
    # else:
    #     draw_object(prostate_obj, x1, y1, z1, 0, 0, 0)


    # No split prostate
    # if pressure >= 0.2 and pressure < 0.5:
    #     # draw_object(prostate_obj, x1, y1, z1, rot_x1, rot_y1, rot_z1)
    #     draw_object(hand_green_obj, x2, y2, z2, rot_x2, rot_y2, rot_z2)
    # elif pressure >= 0.5 and pressure < 1.0:
    #     # draw_object(prostate_obj, x1, y1, z1, rot_x1, rot_y1, rot_z1)
    #     draw_object(hand_yellow_obj, x2, y2, z2, rot_x2, rot_y2, rot_z2)
    # elif pressure >= 1.0 and pressure < 2.0:
    #     # draw_object(prostate_obj, x1, y1, z1, rot_x1, rot_y1, rot_z1)
    #     draw_object(hand_orange_obj, x2, y2, z2, rot_x2, rot_y2, rot_z2)
    # elif pressure >= 2.0:
    #     # draw_object(prostate_obj, x1, y1, z1, rot_x1, rot_y1, rot_z1)
    #     draw_object(hand_red_obj, x2, y2, z2, rot_x2, rot_y2, rot_z2)
    # else:
    #     # draw_object(prostate_obj, x1, y1, z1, rot_x1, rot_y1, rot_z1)
    #     draw_object(hand_obj, x2, y2, z2, rot_x2, rot_y2, rot_z2)

# on the event of resizing the window
@window.event
def on_resize(width, height):  
    global viewport_height, viewport_width, zoom
    viewport_width, viewport_height = window.get_framebuffer_size()
    glViewport(0, 0, viewport_width, viewport_height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glFrustum(-zoom, zoom, -zoom*float(viewport_height)/viewport_width, zoom*float(viewport_height)/viewport_width, 1., 10.)
    glMatrixMode(GL_MODELVIEW)
    return True

@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    global zoom, viewport_height, viewport_width
    if not(zoom <=1 and scroll_y <0):
        zoom += scroll_y/5
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
    pass

@window.event
def on_key_press(symbol, modifiers):
    global cam_pos, rot_cam
    x, y, z= cam_pos
    rot_x, rot_y = rot_cam
    # print(math.sin(math.radians(-rot_x)), math.cos(math.radians(-rot_x)))
    if symbol == key.W:
        x += 1*math.sin(math.radians(-int(rot_x))) 
        z += 1*math.cos(math.radians(-int(rot_x)))            
    elif symbol == key.S:
        x -= 1*math.sin(math.radians(-int(rot_x))) 
        z -= 1*math.cos(math.radians(-int(rot_x)))   
    elif symbol == key.A:
        z -= 1*math.sin(math.radians(-int(rot_x))) 
        x += 1*math.cos(math.radians(-int(rot_x)))  
    elif symbol == key.D:
        z += 1*math.sin(math.radians(-int(rot_x))) 
        x -= 1*math.cos(math.radians(-int(rot_x)))
    elif symbol == key.SPACE or symbol == key.UP:
        y -= 1
    elif symbol == key.LSHIFT or symbol == key.DOWN:
        y += 1
    cam_pos = (x,y,z)
    pass


pyglet.clock.schedule(update)
pyglet.app.run()
sys.exit(app.exec())
#-----------------------------------------------------------------------------------------------------------------------------------------------


