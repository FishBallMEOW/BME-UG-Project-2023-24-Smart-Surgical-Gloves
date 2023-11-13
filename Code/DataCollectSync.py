import serial
import math
from sksurgerynditracker.nditracker import NDITracker
a
#---------------------------------Setup---------------------------------------------------------------------------------------------------------
# set up serial connection to Arduino
sPort_Arduino = serial.Serial('COM6', 9600)  # replace with the correct port and baud rate

#  set up the NDITracker
settings_aurora = { "tracker type": "aurora", "ports to use" : [1,2]}
tracker = NDITracker(settings_aurora)
tracker.use_quaternions = "true"

def ref_sensor():
    """
    Read the data from the aurora using .get_frame() and parse the data into xyz coordinates and orientation in the fomr of quaternions
    Returns:
        ref_pos: a list of the xyz coordinates for the reference sensor
    """
    # Read the data from the sensor 
    data = tracker.get_frame()[3][1]
    # patse the data into position and orientation
    ref_sensor = data[0][4:7]
    
    return ref_sensor

def location_sensor():
    """
    Read the data from the aurora using .get_frame() and parse the data into xyz coordinates and orientation in the fomr of quaternions
    Returns:
        qw : Rotation around vector 
        qx : x quaternion
        qy : y quaternion
        qz : z quaternion
        pos : a list of the xyz coordinates for the location sensor
    """

    # Read the data from the sensor 
    data = tracker.get_frame()[3][0]
    # patse the data into position and orientation
    qw = data[0][0]
    qx = data[0][1]
    qy = data[0][2]
    qz = data[0][3]
    pos = data[0][4:7]

    return qw, qx, qy, qz, pos

def distanceFromRef(ref_pos, pos):
    """
    Calculate the distance between the reference sensor and the location sensor
    
    Arguments:
        ref_pos : A list of the xyz coordinates for the reference sensor
        pos : A list of the xyz coordinate for the location sensor
    Returns:
        distance : the distance between 2 sensors
    """
    distance = math.sqrt((pos[0]-ref_pos[0])**2 + (pos[1]-ref_pos[1])**2 + (pos[2]-ref_pos[2])**2)
#-----------------------------------------------------------------------------------------------------------------------------------------------

#---------------------------------Initialization------------------------------------------------------------------------------------------------
#Initialize aurora
tracker.start_tracking()

#Initialize variables

#-----------------------------------------------------------------------------------------------------------------------------------------------


#---------------------------------Loop----------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------------------------------