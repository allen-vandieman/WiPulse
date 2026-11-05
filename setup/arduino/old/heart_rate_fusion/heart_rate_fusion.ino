/*
  Dual Heart-Rate Sensor Fusion — Arduino Uno R4
  ------------------------------------------------
  Sensors:
    - Gravity Heart Rate Monitor Sensor (SEN0203), analog output -> A0
      Library: DFRobot_Heartrate  (https://github.com/DFRobot/DFRobot_Heartrate)
    - Fermion MAX30102 Heart Rate & Oximeter Sensor (SEN0344 v2), I2C
      Library: DFRobot_BloodOxygen_S  (https://github.com/DFRobot/DFRobot_BloodOxygen_S)

  Behavior:
    Both sensors are sampled continuously and independently. Only when
    BOTH readings are recent AND within AGREEMENT_TOLERANCE_BPM of each
    other does the Arduino send a single line "HR,<bpm>" to the Pi over
    USB serial. Disagreement -> nothing is sent, silently.

  Install both libraries via Arduino Library Manager (search
  "DFRobot_Heartrate" and "DFRobot_BloodOxygen_S") before compiling.
*/

#include <Wire.h>
#include "DFRobot_Heartrate.h"
#include "DFRobot_BloodOxygen_S.h"

// ---------------- SEN0203 (Gravity heart rate, analog) ----------------
#define HR_ANALOG_PIN A0
DFRobot_Heartrate heartrate(ANALOG_MODE);

uint16_t hrPPG = 0;                 // last valid bpm from SEN0203
unsigned long hrPPG_lastUpdate = 0;
unsigned long lastPPGSample = 0;
const unsigned long PPG_SAMPLE_INTERVAL_MS = 20;  // library expects ~20ms sampling cadence

// ---------------- SEN0344 (MAX30102, I2C) ----------------
#define I2C_ADDRESS 0x57
DFRobot_BloodOxygen_S_I2C MAX30102(&Wire, I2C_ADDRESS);

uint16_t hrOx = 0;                  // last valid bpm from MAX30102
unsigned long hrOx_lastUpdate = 0;
unsigned long lastOxPoll = 0;
const unsigned long OX_POLL_INTERVAL_MS = 4000;   // DFRobot's own example: sensor only updates every 4s

// ---------------- Agreement logic ----------------
const uint16_t AGREEMENT_TOLERANCE_BPM = 10;       // how close the two readings must be to "agree" start at 4
const unsigned long MAX_READING_AGE_MS = 10000;   // ignore a reading if it's older than this (>2x the 4s MAX30102 cadence)
const unsigned long MIN_SEND_INTERVAL_MS = 1000;  // don't spam the Pi with duplicate sends
unsigned long lastSendTime = 0;

bool isPlausibleHR(uint16_t bpm) {
  return bpm >= 40 && bpm <= 200;
}

void setup() {
  Serial.begin(115200);   // NOTE: this link also carries the data to the Pi, see comments below
  Wire.begin();
  pinMode(LED_BUILTIN, OUTPUT);

  while (!MAX30102.begin()) {
    // Blink instead of printing, so we don't put junk on the Pi's serial link
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
    delay(500);
  }
  MAX30102.sensorStartCollect();
  digitalWrite(LED_BUILTIN, LOW);
}

void loop() {
  unsigned long now = millis();

  // --- Sample SEN0203 every 20 ms (required by the DFRobot_Heartrate algorithm) ---
  if (now - lastPPGSample >= PPG_SAMPLE_INTERVAL_MS) {
    lastPPGSample = now;
    heartrate.getValue(HR_ANALOG_PIN);
    uint16_t r = heartrate.getRate();   // returns 0 until a fresh 10-beat average is ready
    if (r > 0 && isPlausibleHR(r)) {
      hrPPG = r;
      hrPPG_lastUpdate = now;
    }
  }

  // --- Poll the MAX30102 roughly once a second ---
  if (now - lastOxPoll >= OX_POLL_INTERVAL_MS) {
    lastOxPoll = now;
    MAX30102.getHeartbeatSPO2();
    int r = MAX30102._sHeartbeatSPO2.Heartbeat;
    if (r > 0 && isPlausibleHR((uint16_t)r)) {
      hrOx = (uint16_t)r;
      hrOx_lastUpdate = now;
    }
  }

  // --- Check for agreement and forward to the Pi ---
  bool ppgFresh = (now - hrPPG_lastUpdate) <= MAX_READING_AGE_MS;
  bool oxFresh  = (now - hrOx_lastUpdate)  <= MAX_READING_AGE_MS;

  if (ppgFresh && oxFresh && (now - lastSendTime) >= MIN_SEND_INTERVAL_MS) {
    int diff = (int)hrPPG - (int)hrOx;
    if (abs(diff) <= AGREEMENT_TOLERANCE_BPM) {
      uint16_t agreedHR = (hrPPG + hrOx) / 2;
      Serial.print("HR,");
      Serial.println(agreedHR);
      lastSendTime = now;
    }
  }
}
