# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2014, Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
# vispy: gallery 2

"""
Example demonstrating the use of aspect ratio, and also the flipping
of axis using negative aspect ratios.
"""

import numpy as np

from vispy import app, scene, io

# Read volume
vol1 = np.load(io.load_data_file('volume/stent.npz'))['arr_0']

# Prepare canvas
canvas = scene.SceneCanvas(keys='interactive')
canvas.size = 800, 600
canvas.show()
canvas.measure_fps()

# Set up a viewbox to display the image with interactive pan/zoom
view = canvas.central_widget.add_view()

# Create the volume visuals, only one is visible
volume1 = scene.visuals.Volume(vol1, parent=view.scene, threshold=0.5)

# Create cameras
cam1 = scene.cameras.PanZoomCamera(parent=view.scene)
cam2 = scene.cameras.FlyCamera(parent=view.scene)
cam3 = scene.cameras.TurntableCamera(parent=view.scene)
view.camera = cam3  # Select turntable at first


# Implement key presses
@canvas.events.key_press.connect
def on_key_press(event):
    if event.text == '4':
        cam_toggle = {cam1: cam2, cam2: cam3, cam3: cam1}
        view.camera = cam_toggle.get(view.camera, 'fly')
    elif event.text == '1':
        for cam in (cam1, cam2, cam3):
            flip = cam.flip
            cam.flip = not flip[0], flip[1], flip[2]
    elif event.text == '2':
        for cam in (cam1, cam2, cam3):
            flip = cam.flip
            cam.flip = flip[0], not flip[1], flip[2]
    elif event.text == '3':
        for cam in (cam1, cam2, cam3):
            flip = cam.flip
            cam.flip = flip[0], flip[1], not flip[2]
    elif event.text == '0':
        cam1.set_range()
        cam2.set_range()
        cam3.set_range()

if __name__ == '__main__':
    print(__doc__)
    app.run()