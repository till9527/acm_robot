#include <Motor.h>

Motor motorLeft(12, 6); 
Motor motorRight(8, 5); 

void setup() {
  Serial.begin(9600);
  // Add a small timeout so parseInt() doesn't freeze your loop if it misses data
  Serial.setTimeout(20); 
  delay(3000); 
}

void loop() {
  // Check if data is coming in
  if (Serial.available() > 0) {
    
    // Parse the first number (left speed) and second number (right speed)
    int leftSpeed = Serial.parseInt();
    int rightSpeed = Serial.parseInt();

    // Read the newline character '\n' to finish the command packet
    if (Serial.read() == '\n') {
      
      // Apply Left Motor Speed (Based on your previous forward direction)
      if (leftSpeed > 0) {
        motorLeft.rotateCCW(leftSpeed); 
      } else {
        motorLeft.stop();
      }

      // Apply Right Motor Speed (Based on your previous forward direction)
      if (rightSpeed > 0) {
        motorRight.rotateCW(rightSpeed); 
      } else {
        motorRight.stop();
      }
      
    }
  }
}