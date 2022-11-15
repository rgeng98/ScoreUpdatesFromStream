import numpy as np
import cv2
from mss import mss
from mss import tools
from PIL import Image
import keyboard
import pyautogui
mon = {'left': 85, 'top': 45, 'width': 900, 'height': 450}
r = (85, 45, 900, 450)
global ticker
global count
ticker = 1
count = 200

def grab_ss():
    global ticker
    global count
    screenShot = pyautogui.screenshot(region=r)
    screenShot.save("Train/0/image"+str(count)+".png")
    if ticker == 1:
        ticker = 0
    else:
        ticker = 1
    count=count+1


keyboard.add_hotkey('a', lambda: grab_ss())

print('Starting')
while True:
    pass
