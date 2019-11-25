#!/usr/bin/env python3
"""
    Test harness for dragino module - sends hello world out over LoRaWAN 5 times
"""
import logging
from time import sleep
import RPi.GPIO as GPIO
from dragino import Dragino
import json

GPIO.setwarnings(False)

data = {"action": "sound-detected", "state": True, "battery": 27.0}

D = Dragino("dragino.ini", logging_level=logging.DEBUG)
D.join()
while not D.registered():
    print("Waiting")
    sleep(2)
#sleep(10)

while True:
    D.send(json.dumps(data))
    print("Beacon has been sent!")
    sleep(5)
