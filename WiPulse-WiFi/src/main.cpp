/*
 * ESP32 CSI Traffic Generator with BLE control panel
 * Modified: broadcasting now injects raw 802.11 frames via esp_wifi_80211_tx
 * instead of sending UDP pings — this works even with no associated station,
 * which is required since the Pi drops association once it enters monitor mode.
 */

#include <WiFi.h>
#include "esp_wifi.h"
#include <HardwareBLESerial.h>

// ---- WiFi AP Configuration ----
const char* AP_SSID     = "WiPulse_AP";
const char* AP_PASSWORD = "csi12345";
const int   AP_CHANNEL  = 6;

// ---- State ----
bool wifiOn        = false;
bool broadcasting  = false;
unsigned long broadcastIntervalMs = 100; // default 10 Hz
unsigned long lastPing = 0;
uint32_t seq = 0;

HardwareBLESerial &bleSerial = HardwareBLESerial::getInstance();

// ---------------- Raw 802.11 frame template ----------------
// Minimal broadcast data frame. Addr1 = broadcast, Addr2/Addr3 = ESP32 own MAC.
// Payload carries our seq + timestamp so we can identify/verify captures.
uint8_t rawFrame[128];
const int FRAME_HEADER_LEN = 24; // standard 802.11 MAC header length (no QoS)

void buildFrameTemplate() {
  memset(rawFrame, 0, sizeof(rawFrame));

  // Frame Control: data frame, not protected, to-DS=0, from-DS=0
  rawFrame[0] = 0x08; // type/subtype: data
  rawFrame[1] = 0x00;

  // Duration
  rawFrame[2] = 0x00;
  rawFrame[3] = 0x00;

  // Addr1 = destination = broadcast
  memset(&rawFrame[4], 0xFF, 6);

  // Addr2 = source = ESP32 MAC (filled at send time)
  // Addr3 = BSSID = ESP32 MAC (filled at send time)
  // Seq ctrl
  rawFrame[22] = 0x00;
  rawFrame[23] = 0x00;
}

void sendRawPing() {
  uint64_t sendTime = (uint64_t)esp_timer_get_time();

  // Fill source/BSSID with our own MAC
  uint8_t selfMac[6];
  WiFi.softAPmacAddress(selfMac);
  memcpy(&rawFrame[10], selfMac, 6); // Addr2
  memcpy(&rawFrame[16], selfMac, 6); // Addr3

  // Build payload: seq + timestamp, appended after header
  char payload[48];
  int payloadLen = snprintf(payload, sizeof(payload), "PING,%lu,%llu",
                             (unsigned long)seq, sendTime);

  memcpy(&rawFrame[FRAME_HEADER_LEN], payload, payloadLen);
  int totalLen = FRAME_HEADER_LEN + payloadLen;

  esp_err_t err = esp_wifi_80211_tx(WIFI_IF_AP, rawFrame, totalLen, false);
  if (err != ESP_OK) {
    Serial.printf("esp_wifi_80211_tx failed: %d\n", err);
  }

  seq++;
}

// Forward declarations
void handleCommand(char c);
void applyFrequencyInput(const String &input);
void setWifi(bool on);
void toggleBroadcast();
void printMenu();
void printStatus();
void sendBLE(const String &msg);

// ---------------- Setup / loop ----------------

void setup() {
  Serial.begin(115200);
  delay(300);

  if (!bleSerial.beginAndSetupBLE("WiPulse_CSI_Ctrl")) {
    Serial.println("BLE init failed!");
  } else {
    Serial.println("BLE ready. Connect with a BLE serial app.");
  }

  buildFrameTemplate();
  printMenu();
}

void loop() {
  bleSerial.poll();

  while (bleSerial.available()) {
    char c = bleSerial.read();
    handleCommand(c);
  }

  if (wifiOn && broadcasting) {
    unsigned long now = millis();
    if (now - lastPing >= broadcastIntervalMs) {
      lastPing = now;
      sendRawPing();
    }
  }
}

// ---------------- Command handling ----------------

bool awaitingFreqInput = false;
String freqInputBuffer = "";

void handleCommand(char c) {
  if (awaitingFreqInput) {
    if (c == '\n' || c == '\r') {
      if (freqInputBuffer.length() > 0) {
        applyFrequencyInput(freqInputBuffer);
      }
      freqInputBuffer = "";
      awaitingFreqInput = false;
    } else if ((c >= '0' && c <= '9') || c == '.') {
      freqInputBuffer += c;
    }
    return;
  }

  switch (c) {
    case 'i': case 'I': setWifi(true); break;
    case 'o': case 'O': setWifi(false); break;
    case 'b': case 'B': toggleBroadcast(); break;
    case 'f': case 'F':
      sendBLE("Enter broadcast frequency in Hz (e.g. 10) then press Enter:");
      awaitingFreqInput = true;
      freqInputBuffer = "";
      break;
    case 'p': case 'P': printStatus(); break;
    case 'm': case 'M': case '?': printMenu(); break;
    case '\n': case '\r': break;
    default: sendBLE("Unknown command. Send 'm' for menu."); break;
  }
}

void applyFrequencyInput(const String &input) {
  float hz = input.toFloat();
  if (hz <= 0) { sendBLE("Invalid frequency: " + input); return; }
  if (hz < 0.5) hz = 0.5;
  if (hz > 100) hz = 100;
  broadcastIntervalMs = (unsigned long)(1000.0 / hz);
  if (broadcastIntervalMs < 10) broadcastIntervalMs = 10;
  sendBLE("Broadcast speed set to " + String(hz, 1) + " Hz (interval " + String(broadcastIntervalMs) + " ms)");
}

void setWifi(bool on) {
  if (on == wifiOn) {
    sendBLE(on ? "WiFi already ON" : "WiFi already OFF");
    return;
  }
  if (on) {
    WiFi.mode(WIFI_AP);
    bool ok = WiFi.softAP(AP_SSID, AP_PASSWORD, AP_CHANNEL);
    wifiOn = ok;
    sendBLE(ok ? "WiFi AP started: " + String(AP_SSID) + " ch" + String(AP_CHANNEL)
               : "Failed to start WiFi AP");
  } else {
    broadcasting = false;
    WiFi.softAPdisconnect(true);
    WiFi.mode(WIFI_OFF);
    wifiOn = false;
    sendBLE("WiFi OFF");
  }
}

void toggleBroadcast() {
  if (!wifiOn) {
    sendBLE("Cannot broadcast: WiFi is OFF. Send 'i' to enable WiFi first.");
    return;
  }
  broadcasting = !broadcasting;
  sendBLE(broadcasting ? "Broadcasting STARTED (raw 802.11 injection)" : "Broadcasting STOPPED");
}

void printMenu() {
  sendBLE(
    "\n--- ESP32 CSI Controller ---\n"
    "i : WiFi ON\n"
    "o : WiFi OFF\n"
    "b : Start/Stop raw broadcast ping\n"
    "f : Set broadcast frequency (Hz)\n"
    "p : Print status\n"
    "m : Show this menu\n"
    "-----------------------------\n");
}

void printStatus() {
  float hz = 1000.0 / broadcastIntervalMs;
  String status = "\n--- Status ---\n";
  status += "WiFi: " + String(wifiOn ? "ON" : "OFF") + "\n";
  status += "Broadcasting: " + String(broadcasting ? "YES" : "NO") + "\n";
  status += "Ping speed = " + String(hz, 1) + " Hz (" + String(broadcastIntervalMs) + " ms interval)\n";
  status += "Sequence count: " + String(seq) + "\n";
  status += "--------------\n";
  sendBLE(status);
}

void sendBLE(const String &msg) {
  Serial.println(msg);
  bleSerial.print(msg.c_str());
  bleSerial.print("\n");
}