Eye Tracking System

The Eye-Track System is a Python project that monitors the eyes of the user.
It outlines the eyes and eyelids of the user and identifies the vertical and horizontal position of their eyes, 
indicating whether they are looking up or down or left or right. 
Additionally, a breadboard, equipped with 6 LEDs, will turn on or off depending on the position of the user's eyes.



Libraries used for this project the following:

1) Mediapipe: to collect files of raw data for face-mesh allocation

2) NumPy: to convert the data collected from MediaPipe into coordinates instead of raw numbers

3) OpenCV: to utilize the computer system's camera and to track down the eyelids

4) Serial: to use Arduino UNO to make the 6 LEDS turn on or off according to eye position.
