#include <Servo.h>
#include <ArduinoBLE.h>

Servo servo1;

// ---- Settings (changeable over Bluetooth) ----
int  tDelay   = 1000;   // ms
int  degAngle = 45;     // degrees
double freq = 0.5;       //Hz
bool running  = false;

// ---- Servo timing ----
bool atZero = false;
unsigned long lastMove = 0;

// ---- BLE: Nordic UART Service (works with nRF Connect, Serial Bluetooth Terminal, etc.) ----
BLEService uartService("6E400001-B5A3-F393-E0A9-E50E24DCCA9E");
BLECharacteristic rxChar("6E400002-B5A3-F393-E0A9-E50E24DCCA9E",
                         BLEWrite | BLEWriteWithoutResponse, 20);   // phone -> Arduino
BLECharacteristic txChar("6E400003-B5A3-F393-E0A9-E50E24DCCA9E",
                         BLERead | BLENotify, 20);                  // Arduino -> phone

// ---- Input state machine ----
enum InputState { WAIT_MENU, WAIT_DELAY, WAIT_ANGLE };
InputState inputState = WAIT_MENU;
String lineBuf = "";
bool wasConnected = false;

// ---------- Output helpers ----------
void bleSend(const String &s) {
  // BLE notifications are limited to 20 bytes by default, so send in chunks
  for (unsigned int i = 0; i < s.length(); i += 20) {
    String chunk = s.substring(i, i + 20);
    txChar.writeValue((const uint8_t *)chunk.c_str(), chunk.length());
    delay(10);
  }
}

void out(const String &s) {   // print to both BLE and USB serial
  Serial.print(s);
  if (BLE.connected()) bleSend(s);
}

void printMenu() {
  out("\r\n=== MENU ===\r\n");
  out("1. stop / start motor\r\n");
  out("2. set delay in ms\r\n");
  out("3. set angle in degree\r\n\r\n");
  out("delay = " + String(tDelay) + "ms , Angle = " + String(degAngle) +
      (running ? "  [RUNNING]" : "  [STOPPED]") + "\r\n");
  freq = 1 / ((double)tDelay*2 / 1000);
  out("expected frequency = " + String(freq) + "Hz\r\n");
  out("> ");
}

// ---------- Command handling ----------
void handleLine(String line) {
  line.trim();
  if (line.length() == 0) { printMenu(); return; }

  switch (inputState) {

    case WAIT_MENU:
      if (line == "1") {
        running = !running;
        if (running) lastMove = millis();
        out(running ? "Motor STARTED\r\n" : "Motor STOPPED\r\n");
        printMenu();
      } else if (line == "2") {
        inputState = WAIT_DELAY;
        out("Enter delay in ms (50-10000): ");
      } else if (line == "3") {
        inputState = WAIT_ANGLE;
        out("Enter angle in degrees (0-180): ");
      } else {
        out("Unknown option.\r\n");
        printMenu();
      }
      break;

    case WAIT_DELAY: {
      long v = line.toInt();
      if (v >= 50 && v <= 10000) {
        tDelay = (int)v;
        out("Delay set to " + String(tDelay) + " ms\r\n");
      } else {
        out("Invalid value (50-10000).\r\n");
      }
      inputState = WAIT_MENU;
      printMenu();
      break;
    }

    case WAIT_ANGLE: {
      long v = line.toInt();
      if ((v > 0 || line == "0") && v <= 180) {
        degAngle = (int)v;
        out("Angle set to " + String(degAngle) + " deg\r\n");
        if (running && !atZero) servo1.write(degAngle);  // apply immediately if sitting at the angle
      } else {
        out("Invalid value (0-180).\r\n");
      }
      inputState = WAIT_MENU;
      printMenu();
      break;
    }
  }
}

void pollBle() {
  // New connection -> show the menu
  bool connected = BLE.connected();
  if (connected && !wasConnected) {
    delay(500);   // give the phone time to enable notifications
    inputState = WAIT_MENU;
    lineBuf = "";
    printMenu();
  }
  wasConnected = connected;

  // Incoming data
  if (rxChar.written()) {
    int len = rxChar.valueLength();
    const uint8_t *data = rxChar.value();
    for (int i = 0; i < len; i++) {
      char c = (char)data[i];
      if (c == '\n' || c == '\r') {
        if (lineBuf.length() > 0) {
          handleLine(lineBuf);
          lineBuf = "";
        }
      } else {
        lineBuf += c;
      }
    }
  }
}

// ---------- Setup / loop ----------
void setup() {
  Serial.begin(9600);
  servo1.attach(2);
  servo1.write(0);

  if (!BLE.begin()) {
    Serial.println("Starting BLE failed!");
    while (1);
  }

  BLE.setLocalName("ServoControl");
  BLE.setDeviceName("ServoControl");
  BLE.setAdvertisedService(uartService);
  uartService.addCharacteristic(rxChar);
  uartService.addCharacteristic(txChar);
  BLE.addService(uartService);
  BLE.advertise();

  Serial.println("BLE advertising as 'ServoControl'");
  lastMove = millis();
}

void loop() {
  BLE.poll();
  pollBle();

  // Non-blocking servo sweep (replaces delay())
  if (running && millis() - lastMove >= (unsigned long)tDelay) {
    lastMove = millis();
    atZero = !atZero;
    if (atZero) {
      Serial.println("zero");
      servo1.write(0);
    } else {
      Serial.println(degAngle);
      servo1.write(degAngle);
    }
  }
}