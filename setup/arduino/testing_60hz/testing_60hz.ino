#include <Servo.h>

Servo servo1;

void setup()
{
  Serial.begin(9600);
  servo1.attach(2);
}

void loop() {
  Serial.println("zero");
  servo1.write(0);

  delay(1000);

  Serial.println("45");
  servo1.write(45);

  delay(1000);

}
