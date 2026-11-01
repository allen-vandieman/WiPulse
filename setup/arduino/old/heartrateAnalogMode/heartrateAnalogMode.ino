#define heartPin A0

// Variables for peak detection
int threshold = 475;          // You will likely need to adjust this!
bool isBeating = false;
unsigned long lastBeatTime = 0;

void setup() {
  Serial.begin(115200);
  Serial.println("Heart rate sensor ready. Place your finger on the sensor.");
}

void loop() {
  int heartValue = analogRead(heartPin);
  unsigned long currentTime = millis();

  // If the signal goes above the threshold, we register a beat
  if (heartValue > threshold && !isBeating) {
    isBeating = true;
    
    // Calculate the time since the last beat
    unsigned long ibi = currentTime - lastBeatTime;
    lastBeatTime = currentTime;

    // Filter out unrealistic heart rates (keeps it between ~30 and 200 BPM)
    if (ibi > 300 && ibi < 2000) {
      int bpm = 60000 / ibi;
      
      Serial.print("Raw Value: ");
      Serial.print(heartValue);
      Serial.print(" | BPM: ");
      Serial.println(bpm);
    }
  }

  // Once the signal drops back below the threshold, reset so we can look for the next beat
  if (heartValue < threshold && isBeating) {
    isBeating = false;
  }

  delay(20); // Maintain a ~50Hz sampling rate
}