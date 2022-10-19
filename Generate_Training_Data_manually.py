import numpy as np
import cv2
from mss import mss
from mss import tools
from PIL import Image
import keyboard
mon = {'left': 85, 'top': 45, 'width': 700, 'height': 150}
global ticker
global count
ticker = 1
count = 1
def grab_ss():
    global ticker
    global count
    with mss() as sct:
        screenShot = sct.grab(mon)
        img = Image.frombytes(
            'RGB',
            (screenShot.width, screenShot.height),
            screenShot.rgb,
        )
        tools.to_png(screenShot.rgb, screenShot.size, output = "Train/0/lightning"+str(count)+".png")
        if ticker == 1:
            ticker = 0
        else:
            ticker = 1
        count=count+1


keyboard.add_hotkey(' ', lambda: grab_ss())

while True:
    pass
