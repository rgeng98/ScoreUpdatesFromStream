import torch
import numpy as np
import cv2
from mss import mss
from mss import tools
from torchvision import transforms
from playsound import playsound
from PIL import Image
import time
import matplotlib.pyplot as plt
def SS2Tensor():
    mon = {'left': 85, 'top': 45, 'width': 900, 'height': 450}
    transform = transforms.Compose([transforms.ToTensor(),
                                    # transforms.Normalize(
                                    #            mean=[0.485, 0.456, 0.406],
                                    #            std=[0.229, 0.224, 0.225],
                                    #           )
                                    ])
    # scripted_transforms = torch.jit.script(transform)
    with mss() as sct:
        ss = sct.grab(mon)
        #t = torch.from_numpy(screenShot)
        t = transform(Image.frombytes("RGB", ss.size, ss.bgra,"raw","BGRX"))
    return t

def GOOOOOOOAAAAAAAAALLLLLLLL(sound):
    print("GOOOOAAAAL")
    playsound('GOAL.wav')

sound = "Goal.wav"
model = torch.load("MapleLeafsGoalDetector.pt", map_location=torch.device('cpu'))
while True:
    # Take ss
    t = SS2Tensor()
    with torch.no_grad():
        goal = torch.sigmoid(model(t.unsqueeze(0)))
    print(goal)
    if goal[0][0] > goal[0][1]:
        crit = 0
    else:
        crit = 1
    # crit = goal.detach().numpy()
    # if np.round_(crit, decimals=0) == 0:
    if crit ==0:
        time.sleep(0.25)
    else:
        GOOOOOOOAAAAAAAAALLLLLLLL(sound)

    pass
