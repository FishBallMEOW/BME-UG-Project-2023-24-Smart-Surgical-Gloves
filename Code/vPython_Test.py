from vpython import *
from time import *

scene.width = 1500
scene.height = 600

def display_instructions():
    s = """  To rotate "camera", drag with right button or Ctrl-drag.
  To zoom, drag with middle button or Alt/Option depressed, or use scroll wheel.
    On a two-button mouse, middle is left + right.
  To pan left/right and up/down, Shift-drag.
  Touch screen: pinch/extend to zoom, swipe or two-finger rotate."""
    scene.caption = s

display_instructions()

plate=box(pos=vector(0,-5,0),color=color.white,length=20,width=20,height=0.1)
ball=sphere(pos=vector(0,-5,0),color=color.yellow,radius=1)

while True:
    pass