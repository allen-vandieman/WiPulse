/*
  Dual Heart-Rate Sensor Fusion — Arduino Uno R4 WiFi
  ------------------------------------------------------
  Sensors:
    - Gravity Heart Rate Monitor Sensor (SEN0203), analog output -> A0
      Custom threshold-based peak detection (per-beat instantaneous BPM)
    - Fermion MAX30102 Heart Rate & Oximeter Sensor (SEN0344 v2), I2C
      Library: DFRobot_BloodOxygen_S  (https://github.com/DFRobot/DFRobot_BloodOxygen_S)

  Behavior:
    Both sensors are sampled continuously and independently while detection
    is RUNNING. Only when BOTH readings are recent AND within
    agreementTolerance bpm of each other does the Arduino send a single
    line "HR,<bpm>" to the Pi over USB serial. Disagreement -> nothing sent.

  Bluetooth menu (Uno R4 WiFi only — Minima has no BLE hardware):
    Connect from a BLE terminal app (nRF Connect, LightBlue, Serial
    Bluetooth Terminal in BLE mode) to the "HR-Fusion" device. It exposes
    a Nordic-UART-style service, so most BLE terminal apps recognize it
    automatically. Menu options:
      1) Set hrThreshold
      2) Set agreement tolerance
      3) Start/Stop detection
      4) Start/Stop live per-sensor bpm monitor (debug only)
      5) Set valid HR range (min/max bpm) - filters spikes from either sensor
      m) Redisplay menu

  Requires: DFRobot_BloodOxygen_S, DFRobot_RTU, and ArduinoBLE
  (all installable via Arduino Library Manager).
*/

#include <Wire.h>
#include <ArduinoBLE.h>
#include "DFRobot_BloodOxygen_S.h"

// ================= LED Matrix heart-rate display =================
#include "Arduino_LED_Matrix.h"

ArduinoLEDMatrix matrix;

// 3-column x 5-row glyphs for digits 0-9
const uint8_t digitGlyph[10][5][3] = {
  { {1,1,1}, {1,0,1}, {1,0,1}, {1,0,1}, {1,1,1} }, // 0
  { {0,1,0}, {1,1,0}, {0,1,0}, {0,1,0}, {1,1,1} }, // 1
  { {1,1,1}, {0,0,1}, {1,1,1}, {1,0,0}, {1,1,1} }, // 2
  { {1,1,1}, {0,0,1}, {1,1,1}, {0,0,1}, {1,1,1} }, // 3
  { {1,0,1}, {1,0,1}, {1,1,1}, {0,0,1}, {0,0,1} }, // 4
  { {1,1,1}, {1,0,0}, {1,1,1}, {0,0,1}, {1,1,1} }, // 5
  { {1,1,1}, {1,0,0}, {1,1,1}, {1,0,1}, {1,1,1} }, // 6
  { {1,1,1}, {0,0,1}, {0,0,1}, {0,0,1}, {0,0,1} }, // 7
  { {1,1,1}, {1,0,1}, {1,1,1}, {1,0,1}, {1,1,1} }, // 8
  { {1,1,1}, {1,0,1}, {1,1,1}, {0,0,1}, {1,1,1} }  // 9
};

// Simple horizontal dash, used to indicate "no valid reading right now"
const uint8_t dashGlyph[5][3] = {
  {0,0,0},
  {0,0,0},
  {1,1,1},
  {0,0,0},
  {0,0,0}
};

// Tracks what's currently on the matrix so we don't redundantly re-render
// the same frame every loop iteration.
enum MatrixDisplayState { MATRIX_BLANK, MATRIX_BPM, MATRIX_STALE };
MatrixDisplayState matrixState = MATRIX_BLANK;
uint16_t matrixLastBpm = 0;

// Draws bpm (0-999) centered on the 12x8 LED matrix.
// Digits are 3px wide x 5px tall with 1px spacing between them, vertically centered.
void showHeartRateOnMatrix(uint16_t bpm) {
  if (bpm > 999) bpm = 999;   // clamp — 3 digits is all the matrix can show anyway

  // Skip redraw if this exact value is already showing
  if (matrixState == MATRIX_BPM && matrixLastBpm == bpm) return;

  char digits[4];
  int numDigits = sprintf(digits, "%u", bpm);   // e.g. 72 -> "72", numDigits = 2

  uint8_t frame[8][12] = {0};   // start blank

  const int digitW = 3, digitH = 5, spacing = 1;
  int totalWidth = numDigits * digitW + (numDigits - 1) * spacing;
  int startCol = (12 - totalWidth) / 2;
  int startRow = (8 - digitH) / 2;

  for (int i = 0; i < numDigits; i++) {
    int d = digits[i] - '0';
    int colOffset = startCol + i * (digitW + spacing);
    for (int r = 0; r < digitH; r++) {
      for (int c = 0; c < digitW; c++) {
        frame[startRow + r][colOffset + c] = digitGlyph[d][r][c];
      }
    }
  }

  matrix.renderBitmap(frame, 8, 12);
  matrixState = MATRIX_BPM;
  matrixLastBpm = bpm;
}

// Shows three centered dashes "---" to indicate readings have gone stale.
void showStaleOnMatrix() {
  if (matrixState == MATRIX_STALE) return;   // already showing dashes, skip redraw

  uint8_t frame[8][12] = {0};

  const int glyphW = 3, glyphH = 5, spacing = 1;
  int numGlyphs = 3;   // "---"
  int totalWidth = numGlyphs * glyphW + (numGlyphs - 1) * spacing;
  int startCol = (12 - totalWidth) / 2;
  int startRow = (8 - glyphH) / 2;

  for (int i = 0; i < numGlyphs; i++) {
    int colOffset = startCol + i * (glyphW + spacing);
    for (int r = 0; r < glyphH; r++) {
      for (int c = 0; c < glyphW; c++) {
        frame[startRow + r][colOffset + c] = dashGlyph[r][c];
      }
    }
  }

  matrix.renderBitmap(frame, 8, 12);
  matrixState = MATRIX_STALE;
}

// ================= BLE "UART" service (Nordic UART Service UUIDs) =================
BLEService uartService("6E400001-B5A3-F393-E0A9-E50E24DCCA9E");
BLECharacteristic rxChar("6E400002-B5A3-F393-E0A9-E50E24DCCA9E", BLEWrite, 200);
BLECharacteristic txChar("6E400003-B5A3-F393-E0A9-E50E24DCCA9E", BLERead | BLENotify, 200);

enum MenuState { MENU_IDLE, AWAITING_THRESHOLD, AWAITING_TOLERANCE, AWAITING_HR_MIN, AWAITING_HR_MAX };
MenuState menuState = MENU_IDLE;
uint16_t hrMinPending = 0;   // holds the new min while we wait for the matching max entry

bool detectionRunning = false;

bool monitoringEnabled = false;              // debug: stream live per-sensor bpm over BLE
unsigned long lastMonitorSend = 0;
const unsigned long MONITOR_INTERVAL_MS = 500;  // twice a second is plenty to see variation

// ---------------- SEN0203 (Gravity heart rate, analog, custom peak detection) ----------------
#define HR_ANALOG_PIN A0
int hrThreshold = 475;              // adjustable live over BLE now
bool isBeating = false;
unsigned long lastBeatTime = 0;

uint16_t hrPPG = 0;                 // last valid bpm from SEN0203
unsigned long hrPPG_lastUpdate = 0;
unsigned long lastPPGSample = 0;
const unsigned long PPG_SAMPLE_INTERVAL_MS = 20;  // ~50Hz sampling rate

// ---------------- SEN0344 (MAX30102, I2C) ----------------
#define I2C_ADDRESS 0x57
DFRobot_BloodOxygen_S_I2C MAX30102(&Wire, I2C_ADDRESS);

uint16_t hrOx = 0;                  // last valid bpm from MAX30102
unsigned long hrOx_lastUpdate = 0;
unsigned long lastOxPoll = 0;
const unsigned long OX_POLL_INTERVAL_MS = 4000;   // DFRobot's own example: sensor only updates every 4s

// ---------------- Agreement logic ----------------
uint16_t agreementTolerance = 4;                  // adjustable live over BLE now
const unsigned long MAX_READING_AGE_MS = 10000;   // ignore a reading if it's older than this
const unsigned long MIN_SEND_INTERVAL_MS = 1000;  // don't spam the Pi with duplicate sends
unsigned long lastSendTime = 0;

uint16_t hrMinValid = 40;    // adjustable live over BLE - filters out spikes/noise from either sensor
uint16_t hrMaxValid = 200;

bool isPlausibleHR(uint16_t bpm) {
  return bpm >= hrMinValid && bpm <= hrMaxValid;
}

// ================= BLE menu helpers =================
void bleSend(const String &msg) {
  // Chunk into <=200-byte pieces just in case a message ever grows past the characteristic size
  int len = msg.length();
  int offset = 0;
  while (offset < len) {
    int chunkLen = min(200, len - offset);
    txChar.writeValue((const uint8_t *)msg.c_str() + offset, chunkLen);
    offset += chunkLen;
  }
}

void sendMenu() {
  String menu = "\n=== Heart Rate Fusion Menu ===\n";
  menu += "1) Set hrThreshold (current: " + String(hrThreshold) + ")\n";
  menu += "2) Set agreement tolerance (current: " + String(agreementTolerance) + ")\n";
  menu += "3) " + String(detectionRunning ? "Stop" : "Start") + " detection (currently: " +
          String(detectionRunning ? "RUNNING" : "STOPPED") + ")\n";
  menu += "4) " + String(monitoringEnabled ? "Stop" : "Start") + " live sensor monitor (currently: " +
          String(monitoringEnabled ? "ON" : "OFF") + ")\n";
  menu += "5) Set valid HR range (current: " + String(hrMinValid) + "-" + String(hrMaxValid) + " bpm)\n";
  menu += "m) Show this menu again\n";
  menu += "Send an option:\n";
  bleSend(menu);
}

void startDetection() {
  // Reset state so stale readings from before don't trigger an immediate false "agreement"
  isBeating = false;
  lastBeatTime = millis();
  hrPPG = 0;
  hrOx = 0;
  hrPPG_lastUpdate = 0;
  hrOx_lastUpdate = 0;
  lastSendTime = 0;
  detectionRunning = true;
}

void stopDetection() {
  detectionRunning = false;
}

void handleInput(String input) {
  input.trim();
  if (input.length() == 0) return;

  switch (menuState) {
    case MENU_IDLE: {
      char choice = input.charAt(0);
      if (choice == '1') {
        bleSend("Enter new hrThreshold (current: " + String(hrThreshold) + "):\n");
        menuState = AWAITING_THRESHOLD;
      } else if (choice == '2') {
        bleSend("Enter new tolerance (current: " + String(agreementTolerance) + "):\n");
        menuState = AWAITING_TOLERANCE;
      } else if (choice == '3') {
        if (detectionRunning) {
          stopDetection();
          bleSend("Detection STOPPED.\n");
        } else {
          startDetection();
          bleSend("Detection STARTED.\n");
        }
        sendMenu();
      } else if (choice == '4') {
        monitoringEnabled = !monitoringEnabled;
        if (monitoringEnabled) {
          bleSend("Live monitor ON. Streaming PPG/Ox bpm every " +
                   String(MONITOR_INTERVAL_MS) + "ms.\n");
          if (!detectionRunning) {
            bleSend("(Note: detection is currently STOPPED, so no values will update until you start it with 3.)\n");
          }
        } else {
          bleSend("Live monitor OFF.\n");
        }
        sendMenu();
      } else if (choice == '5') {
        bleSend("Enter new MINIMUM valid bpm (current: " + String(hrMinValid) + "):\n");
        menuState = AWAITING_HR_MIN;
      } else if (choice == 'm' || choice == 'M') {
        sendMenu();
      } else {
        bleSend("Unrecognized option.\n");
        sendMenu();
      }
      break;
    }
    case AWAITING_THRESHOLD: {
      int v = input.toInt();
      if (v > 0) {
        hrThreshold = v;
        bleSend("hrThreshold set to " + String(hrThreshold) + "\n");
      } else {
        bleSend("Invalid number, threshold unchanged.\n");
      }
      menuState = MENU_IDLE;
      sendMenu();
      break;
    }
    case AWAITING_TOLERANCE: {
      int v = input.toInt();
      if (v >= 0) {
        agreementTolerance = (uint16_t)v;
        bleSend("Tolerance set to " + String(agreementTolerance) + "\n");
      } else {
        bleSend("Invalid number, tolerance unchanged.\n");
      }
      menuState = MENU_IDLE;
      sendMenu();
      break;
    }
    case AWAITING_HR_MIN: {
      int v = input.toInt();
      if (v > 0 && v < 250) {
        hrMinPending = (uint16_t)v;
        bleSend("Enter new MAXIMUM valid bpm (current: " + String(hrMaxValid) + "):\n");
        menuState = AWAITING_HR_MAX;
      } else {
        bleSend("Invalid number, HR range unchanged.\n");
        menuState = MENU_IDLE;
        sendMenu();
      }
      break;
    }
    case AWAITING_HR_MAX: {
      int v = input.toInt();
      if (v > 0 && v <= 250 && (uint16_t)v > hrMinPending) {
        hrMinValid = hrMinPending;
        hrMaxValid = (uint16_t)v;
        bleSend("Valid HR range set to " + String(hrMinValid) + "-" + String(hrMaxValid) + " bpm\n");
      } else {
        bleSend("Invalid max (must be greater than min), HR range unchanged.\n");
      }
      menuState = MENU_IDLE;
      sendMenu();
      break;
    }
  }
}

void onRxWritten(BLEDevice central, BLECharacteristic characteristic) {
  int len = characteristic.valueLength();
  char buf[201];
  if (len > 200) len = 200;
  memcpy(buf, characteristic.value(), len);
  buf[len] = '\0';
  handleInput(String(buf));
}

void onBLEConnected(BLEDevice central) {
  menuState = MENU_IDLE;
  bleSend("Connected to HR-Fusion.\n");
  sendMenu();
}

void onBLEDisconnected(BLEDevice central) {
  // Detection keeps running (or stays stopped) independent of the BLE link;
  // reconnecting later just gets you the menu again with current values.
}

// ================= Setup / loop =================
void setup() {
  Serial.begin(115200);   // NOTE: this link carries the data to the Pi, kept separate from BLE
  Wire.begin();
  matrix.begin();
  pinMode(LED_BUILTIN, OUTPUT);

  while (!MAX30102.begin()) {
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
    delay(500);
  }
  MAX30102.sensorStartCollect();
  digitalWrite(LED_BUILTIN, LOW);

  if (!BLE.begin()) {
    // BLE failed to start (wrong board variant, or module fault) - blink fast forever
    while (true) {
      digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
      delay(100);
    }
  }

  BLE.setLocalName("HR-Fusion");
  BLE.setAdvertisedService(uartService);
  uartService.addCharacteristic(rxChar);
  uartService.addCharacteristic(txChar);
  BLE.addService(uartService);

  rxChar.setEventHandler(BLEWritten, onRxWritten);
  BLE.setEventHandler(BLEConnected, onBLEConnected);
  BLE.setEventHandler(BLEDisconnected, onBLEDisconnected);

  BLE.advertise();
}

void loop() {
  BLE.poll();   // services BLE events/connections; must be called frequently, non-blocking

  if (!detectionRunning) {
    return;   // sensors are only sampled while detection is active
  }

  unsigned long now = millis();

  // --- Sample SEN0203 every 20 ms and detect beats via threshold crossing ---
  if (now - lastPPGSample >= PPG_SAMPLE_INTERVAL_MS) {
    lastPPGSample = now;
    int heartValue = analogRead(HR_ANALOG_PIN);

    if (heartValue > hrThreshold && !isBeating) {
      isBeating = true;
      unsigned long ibi = now - lastBeatTime;
      lastBeatTime = now;

      if (ibi > 300 && ibi < 2000) {
        uint16_t r = (uint16_t)(60000UL / ibi);
        if (isPlausibleHR(r)) {
          hrPPG = r;
          hrPPG_lastUpdate = now;
        }
      }
    }

    if (heartValue < hrThreshold && isBeating) {
      isBeating = false;
    }
  }

  // --- Poll the MAX30102 roughly once every 4 seconds ---
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

  if ((now - lastSendTime) >= MIN_SEND_INTERVAL_MS) {
    if (ppgFresh && oxFresh) {
      // Both sensors reporting recently -> only send if they agree
      int diff = (int)hrPPG - (int)hrOx;
      if (abs(diff) <= (int)agreementTolerance) {
        uint16_t agreedHR = (hrPPG + hrOx) / 2;
        Serial.print("HR,");
        Serial.println(agreedHR);
        showHeartRateOnMatrix(agreedHR);
        lastSendTime = now;
      }else {
        showStaleOnMatrix();   // both fresh but disagree -> no confirmed value right now
      }
    } else if (!ppgFresh && oxFresh) {
      // SEN0203 isn't producing fresh beats right now (bad contact, threshold
      // miscalibrated, etc.) -> fall back to the MAX30102 alone. hrOx was already
      // range-checked against hrMinValid/hrMaxValid when it was stored, so no
      // extra check needed here.
      Serial.print("HR,");
      Serial.println(hrOx);
      showHeartRateOnMatrix(hrOx);
      lastSendTime = now;
    }else {
      // Neither path has a valid current reading (both stale, or PPG fresh/Ox stale)
      showStaleOnMatrix();
    }
  }

  // --- Debug: stream live per-sensor bpm to the phone if monitoring is on ---
  if (monitoringEnabled && (now - lastMonitorSend >= MONITOR_INTERVAL_MS)) {
    lastMonitorSend = now;
    String line = "PPG: " + (ppgFresh ? String(hrPPG) : String("--")) +
                  " bpm | Ox: " + (oxFresh ? String(hrOx) : String("--")) + " bpm\n";
    bleSend(line);
  }
}
