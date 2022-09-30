#!/bin/bash

USBARG=$(ls -l /dev/ttyACM*)
echo "${USBARG}"

sudo chmod 777 /dev/ttyACM0
rosrun rosserial_python serial_node.py /dev/ttyACM0
