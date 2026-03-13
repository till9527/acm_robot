/*
  Motor.h - Library for driving a dc motor using the Adafruit Adafruit TB6612 motor driver board.
  Link to the motor driver board: 
  https://learn.adafruit.com/adafruit-tb6612-h-bridge-dc-stepper-motor-driver-breakout/overview
  
  Created by Li Dang and Girma Tewolde, October 13, 2015.
  Last modified on February 2, 2025.
  Released into the public domain.
*/
#ifndef Motor_h
#define Motor_h

#include "Arduino.h"

class Motor
{
  public:
    Motor(int d_pin, int s_pin);	//initialize the motor object with direction & speed pins
	//note that the directions CW & CCW assume looking into the motor shaft from the outside
    void rotateCW(int speed);		//rotate motor clockwise at the given speed (0 to 100)
    void rotateCCW(int speed);		//rotate motor counter-clockwise at the given speed (0 to 100)
	void stop();					//stop motot
  private:
    int dir_pin, speed_pin;			//define the motor pins
};

#endif