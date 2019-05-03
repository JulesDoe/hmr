import numpy as np
import cv2
import time



cap = cv2.VideoCapture(0)




while(True):
    
    cap.get(3)

    # ret = cap.set(3,320)
    # ret = cap.set(4,240)

    # time.sleep(2)

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    P = np.random.rand(100,3)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()