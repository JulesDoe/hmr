import numpy as np
import cv2

import vispy
import vispy.scene
from vispy.scene import visuals



# Video capture
cap = cv2.VideoCapture(0)


# Make a canvas and add simple view
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

# create scatter object
scatter = visuals.Markers()

# generate data or figure out how to prevent crash without data ^^
pos = np.random.normal(size=(100000, 3), scale=0.2)
scatter.set_data(pos, edge_color=None, face_color=(1, 1, 1, .5), size=5)

view.add(scatter)

#configure view
view.camera = 'turntable'  # or try 'arcball'
axis = visuals.XYZAxis(parent=view.scene)


while(True):
    
    cap.get(3)

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    P = np.random.rand(100,3)

    # Display Camera frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Display Plot
    pos = np.random.normal(size=(100000, 3), scale=0.2)
    scatter.set_data(pos, edge_color=None, face_color=(1, 1, 1, .5), size=5)




# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()