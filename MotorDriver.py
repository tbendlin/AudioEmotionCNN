"""
    File that takes care of driving Furby's motor
"""

# Import required modules
import time
import RPi.GPIO as GPIO

STBY = 13
AIN1 = 16
AIN2 = 15
PWMA = 7

class MotorDriver:
    def __init__(self):
        # Declare the GPIO settings
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        
        # set up GPIO pins
        GPIO.setup(PWMA, GPIO.OUT) # Connected to PWMA
        GPIO.setup(AIN2, GPIO.OUT) # Connected to AIN2
        GPIO.setup(AIN1, GPIO.OUT) # Connected to AIN1
        GPIO.setup(STBY, GPIO.OUT) # Connected to STBY
    
    def __start_motor__(self):
        # Drive the motor
        GPIO.output(AIN1, GPIO.HIGH)
        GPIO.output(AIN2, GPIO.LOW)

        # Set the motor speed
        GPIO.output(PWMA, GPIO.HIGH)
        
        # Disable the STBY
        GPIO.output(STBY, GPIO.HIGH)
    
    def __stop_motor__(self):
        # Reset all the GPIO pins by setting them to LOW
        GPIO.output(AIN1, GPIO.LOW) # Set AIN1
        GPIO.output(AIN2, GPIO.LOW) # Set AIN2
        GPIO.output(PWMA, GPIO.LOW) # Set PWMA
        GPIO.output(STBY, GPIO.LOW) # Set STBY
