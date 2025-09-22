# CV_FILTERS

How to Run the Live Camera Effects Application
This guide will walk you through the steps to set up and run the camera_effects_with_panorama.py script.

1. Prerequisites
Before running the application, you need to have Python and several libraries installed.

Install Python
If you don't have Python, download and install it from python.org.

Install Required Libraries
Open your terminal or command prompt and run the following command to install the necessary computer vision libraries. The opencv-contrib-python package is important as it includes the SIFT algorithm used for panorama stitching and the ArUco module for augmented reality.

pip install opencv-contrib-python numpy

2. Required Files
For all features to work correctly, you need a couple of additional files in the same directory as the Python script:

trex_model.obj: A 3D model file required for the Augmented Reality mode. You can find sample .obj files online to use.

calibration.npz: A camera calibration file. The application can generate this file for you. It's necessary for accurate augmented reality and lens undistortion.

3. Running the Application
Once the prerequisites are met, you can run the application with a simple command:

Navigate to the directory containing the script and required files in your terminal.

Run the script using Python:

python camera_effects_with_panorama.py

A window titled "Live Camera Effects" should appear, showing your live webcam feed.

4. How to Use the Features
The application has several modes and controls, which are listed on the sidebar of the application window.

Basic Controls
Use the number and letter keys as shown in the on-screen menu to switch between different filters and modes (e.g., 1 for Grayscale, g for Gaussian Blur).

Adjust contrast with c and v.

Adjust brightness with b and n.

Camera Calibration (Key: k)
This mode is required for AR and Undistort to work well.

Print out a chessboard pattern (9x6 squares).

Press k to enter calibration mode.

Show the chessboard to the camera from different angles and distances. The application will automatically capture images when it detects the board.

Once it captures 20 images, it will perform the calibration and save the data to calibration.npz.

Panorama Stitching (Keys: o, p, z)
Point your camera at the starting scene of your panorama and press o to capture the first frame.

Pan your camera to the right, ensuring about 40-50% overlap with the previous view, and press o again.

Repeat this process for as many frames as you need.

Press p to start the stitching process. This may take a few moments.

Once complete, the final panorama will be displayed.

Press z to reset and start a new panorama.

Augmented Reality (Key: a)
Make sure you have a calibration.npz file and a trex_model.obj file in the directory.

You will also need an ArUco marker. You can generate and print one online.

Press a to enter AR mode.

Show the ArUco marker to the camera. The 3D model should appear overlaid on the marker.
