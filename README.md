**Running Raspberry Pi code remotely:**

If this is the first time this specific Pi has ever been used to SSH, make sure that SSH is enabled. On the raspberry pi, go to a terminal and type “sudo raspi-config”. Go to interface options -> SSH -> yes. Then click on “finish”. You will also want to make sure that you enable serial ports similarly, since this code relies on being able to communicate with the arduino via USB

Then, find the IP address of your Pi. Type “hostname -I”

Then in putty, make a new profile that has that IP address as its connection

Then connect to the pi using your username and password

Once you’re in, assuming you followed the [instructions](https://github.com/till9527/Raspberry_PI_Models) to make a venv for the ai camera, type “source ai_cam/bin/activate” to enter the virtual environment needed

Before running, make sure to flash the DriveRobotExample_ino3.ino code found inside of the motor library onto the uno R4. Make sure that on the computer you’re flashing the code, you put motor into the arduino libraries folder (C:\Users\user\Documents\Arduino\libraries)

Then do “cd Desktop/acm_robot”, and then “python run_robot.py” (at least assuming that you cloned this repository directly onto your desktop)
