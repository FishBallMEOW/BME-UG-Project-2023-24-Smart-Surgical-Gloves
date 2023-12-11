import ctypes
import os
import pyglet 
from pyglet.gl import *
from pyglet.window import mouse
from pywavefront import visualization, Wavefront
import math

root_path = os.path.dirname(__file__)
hand_obj = Wavefront(os.path.join(root_path, 'hand/hand.obj'))

viewport_width=1280
viewport_height=720
rotation = 0.0
timer = 0.0
lightfv = ctypes.c_float * 4
x1 = 0.0
y1 = 0.0
z1 = -30.0
steps = -1


# Creating a window
window = pyglet.window.Window(viewport_width, viewport_height, "Surgical Gloves", resizable=True)

# on the event of creating/ drawing the window
@window.event  
def on_draw():  
    window.clear()
    glLoadIdentity()
    glLightfv(GL_LIGHT0, GL_POSITION, lightfv(-1.0, 1.0, 1.0, 0.0))
    draw_object(hand_obj, x1, y1, z1, 60, -90, 0)

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

# on mouse motion event
@window.event
def on_mouse_motion(x, y, dx, dy):
    global x1, y1, z1, viewport_width, viewport_height
    x1 = -(2*x*z1/viewport_width - z1)
    y1 = -(2*y*z1*math.tan(22.5)/viewport_height - z1*math.tan(22.5))

def draw_object(obj, x, y, z, rot_x, rot_y, rot_z):
    glLoadIdentity()
    glTranslated(x, y, z)
    glRotatef(rot_y, 0.0, 1.0, 0.0)
    glRotatef(rot_x, 1.0, 0.0, 0.0)
    glRotatef(rot_z, 0.0, 0.0, 1.0)
    visualization.draw(obj)


def update(dt):
    global x1, y1, z1, timer, rotation, steps
    if z1 == -100:
        steps = 1
    elif z1 == -10:
        steps = -1
    z1=z1+steps
    rotation += 90.0 * dt
    if rotation > 720.0:
        rotation = 0.0
    print(x1, y1)
    
    

# Run the function update every
pyglet.clock.schedule(update)

# Run the app
pyglet.app.run()
