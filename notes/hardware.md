# Hardware Setup

An Arduino and two sensors was used to obtain groundtruth data

## Fermion: MAX30102 Heart Rate and Oximeter Sensor

DFRobot Heart Rate and Oximeter Sensor Breakout Board Heart Rate Sensor Breakout Board for Sensor

### BUY

https://za.rs-online.com/web/p/sensor-development-tools/2473241

### Information

https://wiki.dfrobot.com/sen0344/

### Arduido Code

```
gainHeartbearSPO2

/LIBS
|
|-DFRobot_BloodOxygen_S
|-DFRobot_RTU
```

## Gravity: Heart Rate Monitor Sensor

https://wiki.dfrobot.com/sen0203

### Arduido Code

```
heartrateAnalogMode

/LIBS
|
|-DFRobot_Heartrate
```

## Arduino Uno R4 Wifi

```
# Test both sensors with Arduino
heart_rate_fusion

# V2 including bluetooth feature
heart_rate_fusion-bluetooth_V2

# V3 including min/max
heart_rate_fusion-bluetooth_V3

/LIBS
|
|-DFRobot_BloodOxygen_S
|-DFRobot_RTU
|-DFRobot_Heartrate
|-ArduinoBLE
|-HardwareBLESerial
```

### Setup

Connect Arduino Uno R4 USB port to any USB-A port on the Pi 5

 - Find the serial device name

```bash
# before and after plugging in the Arduino
ls /dev/tty* 
```

Usually something like

`/dev/ttyACM0` or `/dev/ttyACM1`

Give your user permission to use it

```bash
sudo usermod -aG dialout $USER
# log out and back in (or reboot) for it to take effect. 
```
Sanity-check

```bash
# install screen if needed
sudo apt install screen 

# Sanity-check the raw link
screen /dev/ttyACM0 115200

# Ctrl-A then K to exit screen.
```

### Install pyserial and run the reader script

```bash
pip3 install pyserial

python3 pi_serial_recorder.py
# If the device wasn't `/dev/ttyACM0`, edit the `PORT` variable at the top of the script first.
```