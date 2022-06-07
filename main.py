import screenshot
import det
import op

import time 

if __name__ == '__main__':
    while(1):
        start = time.time()
        img = screenshot.shot()
        # img.show()
        dir = det.detection(img)
        op.direct(dir)
        end = time.time()
        print(dir,end-start)
