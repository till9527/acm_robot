#include "Arduino.h"
#include "Motor.h"

Motor::Motor(int d_pin, int s_pin)
{
  pinMode(d_pin, OUTPUT);
  pinMode(s_pin, OUTPUT);
  dir_pin = d_pin;
  speed_pin = s_pin;
}

//motor speed input is in the range 0..100
void Motor::rotateCW(int speed)
{
	digitalWrite(dir_pin, LOW);
	analogWrite(speed_pin, map(speed, 0, 100, 0, 255));
}


void Motor::rotateCCW(int speed)
{
	digitalWrite(dir_pin, HIGH);
	analogWrite(speed_pin, map(speed, 0, 100, 0, 255));
}

void Motor::stop()
{
	digitalWrite(dir_pin, LOW);
	analogWrite(speed_pin, 0);
}