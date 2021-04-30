import time
from fractions import Fraction

from picamera import PiCamera
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--iso", dest='iso', type=int, help="iso")
parser.add_argument("--shutter", dest='shutter_speed', type=int, help="shutterspeed")
parser.add_argument("--contrast", dest='contrast', type=int, help="contrast")
parser.add_argument("--interval", dest='interval', type=int, help="interval")
args = parser.parse_args()

iso = args.iso
shutter_speed = args.shutter_speed
contrast = args.contrast
interval = args.interval
resolution = (2592, 1952)

with PiCamera(resolution=resolution, framerate=Fraction(1000000, shutter_speed)) as camera:
    camera.iso = iso
    camera.shutter_speed = shutter_speed
    camera.contrast = contrast
    iterator = 0
    time.sleep(2)

    while True:
        filename = f'data/{str(iterator)}.jpg'
        with open(filename, "wb") as file:
            frame = camera.capture(file, format="jpeg", use_video_port=False)
        iterator += 1
        print("snap")

        time.sleep(interval)


