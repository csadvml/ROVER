import argparse
import shutil
import base64
from datetime import datetime
import os
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO, StringIO
import json
import pickle
import matplotlib.image as mpimg
import time
import asyncio
import logging

from perception import perception_step
from decision import decision_step
from supporting_functions import update_rover, create_output_images

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize socketio server and Flask application 
sio = socketio.Server()
app = Flask(__name__)

# Read in ground truth map and create 3-channel green version for overplotting
ground_truth = mpimg.imread('../calibration_images/map_bw.png')
ground_truth_3d = np.dstack((ground_truth*0, ground_truth*255, ground_truth*0)).astype(np.float16)

# Define RoverState() class to retain rover state parameters
class RoverState():
    def __init__(self):
        self.start_time = None
        self.total_time = None
        self.stuck_time = 0
        self.rock_time = 0
        self.img = None
        self.pos = None
        self.yaw = None
        self.pitch = None
        self.roll = None
        self.vel = None
        self.steer = 0
        self.throttle = 0
        self.brake = 0
        self.nav_angles = None
        self.nav_dists = None
        self.samples_angles = None
        self.samples_dists = None
        self.ground_truth = ground_truth_3d
        self.mode = ['forward']
        self.throttle_set = 0.5
        self.brake_set = 10
        self.stop_forward = 100
        self.go_forward = 500
        self.max_vel = 3
        self.vision_image = np.zeros((160, 320, 3), dtype=np.float16)
        self.worldmap = np.zeros((200, 200, 3), dtype=np.float16)
        self.samples_pos = None
        self.samples_to_find = 0
        self.samples_located = 0
        self.samples_collected = 0
        self.near_sample = 0
        self.picking_up = 0
        self.send_pickup = False

Rover = RoverState()

# Define frame counters and FPS calculation variables
frame_counter = 0
second_counter = time.time()
fps = None

# Telemetry function to handle incoming data
@sio.on('telemetry')
async def telemetry(sid, data):
    global frame_counter, second_counter, fps
    frame_counter += 1
    if (time.time() - second_counter) > 1:
        fps = frame_counter / (time.time() - second_counter)
        frame_counter = 0
        second_counter = time.time()
    logger.info(f"Current FPS: {fps:.2f}")

    if data:
        global Rover
        Rover, image = update_rover(Rover, data)

        if np.isfinite(Rover.vel):
            Rover = perception_step(Rover)
            Rover = decision_step(Rover)

            out_image_string1, out_image_string2 = create_output_images(Rover)

            if Rover.send_pickup and not Rover.picking_up:
                send_pickup()
                Rover.send_pickup = False
            else:
                commands = (Rover.throttle, Rover.brake, Rover.steer)
                send_control(commands, out_image_string1, out_image_string2)
        else:
            send_control((0, 0, 0), '', '')

        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save(f'{image_filename}.jpg')
    else:
        sio.emit('manual', data={}, skip_sid=True)

# Function to send control commands
def send_control(commands, image_string1, image_string2):
    data = {
        'throttle': commands[0].__str__(),
        'brake': commands[1].__str__(),
        'steering_angle': commands[2].__str__(),
        'inset_image1': image_string1,
        'inset_image2': image_string2,
    }
    sio.emit("data", data, skip_sid=True)
    eventlet.sleep(0)

# Function to send pickup command
def send_pickup():
    logger.info("Picking up")
    pickup = {}
    sio.emit("pickup", pickup, skip_sid=True)
    eventlet.sleep(0)

# SocketIO connection event
@sio.on('connect')
def connect(sid, environ):
    logger.info(f"Connected with sid: {sid}")
    send_control((0, 0, 0), '', '')
    sample_data = {}
    sio.emit("get_samples", sample_data, skip_sid=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('image_folder', type=str, nargs='?', default='', help='Path to image folder. This is where the images from the run will be saved.')
    args = parser.parse_args()

    if args.image_folder != '':
        logger.info(f"Creating image folder at {args.image_folder}")
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        logger.info("Recording this run ...")
    else:
        logger.info("NOT recording this run ...")

    app = socketio.Middleware(sio, app)

    logger.info("Starting Eventlet server on port 4567...")
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
