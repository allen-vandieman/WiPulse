#include <Arduino.h>
#include <HardwareBLESerial.h>

HardwareBLESerial &bleSerial = HardwareBLESerial::getInstance();

String msg;

void setup() {
  Serial.begin(115200);
  if (!bleSerial.beginAndSetupBLE("WiPulse-Serial")) {
    Serial.println("BLE failed to start!");
    while (1);
  }

  Serial.println("BLE UART ready. Waiting for connection...");

  while (!bleSerial) {
      bleSerial.poll();
      delay(10);
    }
  
  while(!bleSerial.available())
    {
      bleSerial.poll();
      msg = "Press Enter to start!\n";
      bleSerial.print(msg.c_str());
    }

    Serial.println("BLE device connected!");
}

void loop() {
  bleSerial.poll();
  msg = "****\nMenu\n****\n1. Use Potentiometer\n2. Use Sensor\n3. Use Phone\n4. Led on\n5. Led off\n6. Update Max for sensor\n7. Update Difference for sensor\n"; //String((millis()/1000)) + "\n";
  bleSerial.print(msg.c_str());   

  while(!bleSerial.available())
  {
    bleSerial.poll();
  }




  // Receive data from phone
  while (bleSerial.available()) {
    char c = bleSerial.read();
    char t = '0';
    Serial.print("Received: ");
    Serial.println(c);

    msg = "Press any key to exit selection\n";
    bleSerial.print(msg.c_str());

    switch(c)
    {
      case '1':
        
        while(!bleSerial.available())
        {
          bleSerial.poll();
          //do stuff and wait for exit command!
        }
        break;

      case '2':
        
        while(!bleSerial.available())
        {
          bleSerial.poll();
          //do stuff and wait for exit command!
        }
        break;

      case '3':
        while(t!='4')
        {
          msg = "-------\nOptions\n-------\n0. Off\n1. Slow\n2. Medium\n3. Max\n4. exit\n"; 
          bleSerial.print(msg.c_str());

          while(!bleSerial.available())
          {
            bleSerial.poll();
          }

          t = bleSerial.read();
          switch(t)
          {
            case '0':
              //sub job
              break;
            case '1':
              //sub job
              break;
            case '2':
              //sub job
              break;
            case '3':
              //sub job
              break;
            case '4':
              msg = "Leaving menu\n";
              bleSerial.print(msg.c_str());
              break;
            default:
              msg = "Invalid option selected!\n";
              bleSerial.print(msg.c_str());
          }
           
        }   
        break;

      case '4':
        //do work and continue;
        break;
      case '5':
        //do work and continue;
        break;
      case '6':
        //do work and continue;
        break;
      case '7':
        //do work and continue;
        break;
      default:
      msg = "Invalid option selected!\n";
        bleSerial.print(msg.c_str());
    }

  }
}

