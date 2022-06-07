import pyautogui as a
from time import sleep
from random import randint as rand

spd = "neutral" # drive, brake, neutral, reverse
dir = "mid" # left, right, mid

spdLs = ['drive','neutral', 'brake']
dirLs = ['left', 'right', 'mid']

# 速度控制
def speed(state):
    if state == "neutral":
        a.keyUp("up")
        a.keyUp("down")
    elif state == "drive":
        a.keyDown("up")
        a.keyUp("down")
    elif state == "brake":
        a.keyUp("up")
        a.keyDown("down")
        
# 方向控制
def direct(state):
    if state == "mid":
        a.keyUp("left")
        a.keyUp("right")
    elif state == "left":
        a.keyDown("left")
        a.keyUp("right")
    elif state == "right":
        a.keyUp("left")
        a.keyDown("right")

# 测试
def keyboradTest():
    ntime = rand(1,3)
    nspd = spdLs[rand(1,3)-1]
    ndir = dirLs[rand(1,3)-1]
    print(nspd,ndir,ntime)
    speed(nspd)
    direct(ndir)
    sleep(ntime)

def ctrl_test():
    for i in range(5):
        print(i)
        sleep(1)
    print("start")

    while(1):
        ntime = rand(1,3)
        ndir = dirLs[rand(0,2)]
        nspd = spdLs[rand(1,3)-1]
        print(nspd,ndir,ntime)
        speed(nspd)
        direct(ndir)        
        sleep(ntime)  
        # pass
        # 识别
        # 判断
        # 控制






