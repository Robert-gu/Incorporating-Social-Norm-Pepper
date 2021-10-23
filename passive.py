#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

"""Example: Use findFreeZone Method"""

import Naoqi
import argparse
import sys
import math
import time


def main(session):
    """
    This example uses the moveTo method.
    """

    motion_service  = session.service("ALMotion")
    posture_service = session.service("ALRobotPosture")
    asr_service = session.service("ALAnimatedSpeech")
    leds_service = session.service("ALLeds")
    configuration = {"bodyLanguageMode":"contextual"}

    motion_service.wakeUp()

    duration = 1.0
    leds_service.rasta(duration)

    motion_service.moveTo(2, 0, 0)
    time.sleep(1)
    asr_service.say("Hello, I am Pepper. ^start(animations/Stand/Gestures/Hey_1) I am here to help!", configuration)
    time.sleep(1)
    motion_service.moveTo(1, 0, 0)
    time.sleep(1)
    asr_service.say("Please calm down. ^start(animations/Stand/Gestures/CalmDown_1) I am here to help.", configuration)
    time.sleep(1)
    motion_service.moveTo(0.5, 0, 0)
    time.sleep(1)
    asr_service.say("Please use my tablet to access useful information. ^start(animations/Stand/Gestures/ShowTablet_2) ", configuration)
    time.sleep(1)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    main(session)