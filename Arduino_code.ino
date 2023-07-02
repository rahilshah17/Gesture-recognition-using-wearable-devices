#include <Arduino_LSM6DS3.h>

float Ax, Ay, Az;//Accelerometer readings
float Gx, Gy, Gz;//Gyroscope readings
unsigned long ms_from_start;
unsigned long ms_prev = 0;
unsigned long interval = 100;//delay of 100ms

//Setup
void setup() {
  Serial.begin(9600);

  while(!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println("Hz");
  Serial.println();

  Serial.print("Gyroscope sample rate = ");  
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println("Hz");
  Serial.println();

}

void loop() {

  ms_from_start = millis();
  if (ms_from_start-ms_prev>=interval){
    ms_prev = ms_from_start;
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      //Reading and printing the sensor values
      IMU.readAcceleration(Ax, Ay, Az);
      IMU.readGyroscope(Gx, Gy, Gz);
      Serial.print(Ax);
      Serial.print(',');
      Serial.print(Ay);
      Serial.print(',');  
      Serial.print(Az);
      Serial.print(',');
      Serial.print(Gx);
      Serial.print(',');
      Serial.print(Gy);
      Serial.print(',');
      Serial.print(Gz);
      Serial.println();
    }
  }

// delay(100);
// data is printed every 0.1 s
}
