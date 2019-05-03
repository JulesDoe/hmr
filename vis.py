# -*- coding: utf-8 -*-
# vispy: gallery 10
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

""" Demonstrates use of visual.Markers to create a point cloud with a
standard turntable camera to fly around with and a centered 3D Axis.
"""

import numpy as np
import vispy.scene
from vispy.scene import visuals
from vispy import app, scene

#
# Make a canvas and add simple view
#
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()




# create scatter object and fill in the data
scatter = visuals.Markers()
# generate data
pos = np.random.normal(size=(100000, 3), scale=0.2)
scatter.set_data(pos, edge_color=None, face_color=(1, 1, 1, .5), size=5)

view.add(scatter)

view.camera = 'turntable'  # or try 'arcball'

# add a colored 3D axis for orientation
axis = visuals.XYZAxis(parent=view.scene)


def update(ev):
    # generate data
    pos = np.random.normal(size=(100000, 3), scale=0.2)
    scatter.set_data(pos, edge_color=None, face_color=(1, 1, 1, .5), size=5)




timer = app.Timer()
timer.connect(update)
timer.start(0.05)  # interval, iterations

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        vispy.app.run()