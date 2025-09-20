import cv2
import time
from picamera2 import Picamera2
import numpy as np


frame = None
Matrix = None
framePers = None
frameGray = None
frameThresh = None
frameEdge = None
frameFinal = None
frameFinalDuplicate = None
ROILane = None
LeftLanePos = None
RightLanePos = None
frameCenter = None
laneCenter = None
Result = None
ss = None
histogramLane = None
frame_stop = None
roi_stop = None
gray_stop = None
dist_stop = 0

# Initialize camera object
camera = Picamera2()
# Configure camera preview settings
camera.configure(camera.create_preview_configuration(main={"size": (400, 240)}))
# Start capturing frames
camera.start()

# Source variable is an array representing the region of interest that we 
# want to isolate from the background, roughly representing the area where the road lines may be found.
Source = np.array([(30, 145), (370, 145), (0, 195), (400, 195)], dtype=np.float32)
# Destination variable represents points used for perspective transformation.
Destination = np.array([(100, 0), (280, 0), (100, 240), (280, 240)], dtype=np.float32)

def Capture():
	"""
	Function to capture a frame from the camera and preprocess it
	"""
	# Here we access global variables
	global frame
	global frame_stop
	# Here we capture a frame from the camera
	frame = camera.capture_array()
	# and convert the captured frame to RGB format
	frame_stop = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	# Again here we convert the captured frame to RGB format and store it in the global variable 'frame' (this is the main frame used for lane detection processing).
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def Perspective():
	# Function to perform perspective transformation
    
    # Access global variables
    global frame, Matrix, framePers
	# Here we draw lines on the frame to represent the region of interest (ROI). In our case, the region of interest is a trapezoid-shaped polygon.
    cv2.line(frame, tuple(Source[0].astype(int)), tuple(Source[1].astype(int)), (0, 0, 255), 2)
    cv2.line(frame, tuple(Source[1].astype(int)), tuple(Source[3].astype(int)), (0, 0, 255), 2)
    cv2.line(frame, tuple(Source[3].astype(int)), tuple(Source[2].astype(int)), (0, 0, 255), 2)
    cv2.line(frame, tuple(Source[2].astype(int)), tuple(Source[0].astype(int)), (0, 0, 255), 2)

	# The following function calculates the perspective transformation matrix using the source and destination points.
    Matrix = cv2.getPerspectiveTransform(Source, Destination)
	# The perspective transformation essentially warps the ROI defined by the source points to a rectangular shape defined by the destination points, 
	# allowing for a bird's eye view perspective.
    framePers = cv2.warpPerspective(frame, Matrix, (400, 240))




def Threshold():
	# This function, named Threshold, is responsible for performing thresholding and edge detection on the perspective-transformed frame (framePers).
	
    global framePers, frameGray, frameThresh, frameEdge, frameFinal, frameFinalDuplicate
	# Here we convert the perspective-transformed frame (framePers) to grayscale (frameGray) using cv2.cvtColor()
    frameGray = cv2.cvtColor(framePers, cv2.COLOR_RGB2GRAY)
	# Here it is applied thresholding to the grayscale image (frameGray) to create a binary image (frameThresh) using cv2.threshold().
    # Pixels with intensities greater than 200 are set to 255 (white), and all other pixels are set to 0 (black).
    _, frameThresh = cv2.threshold(frameGray, 200, 255, cv2.THRESH_BINARY)
	# Then we apply edge detection on the grayscale image (frameGray) using the Canny edge detection algorithm
    frameEdge = cv2.Canny(frameGray, 400, 800)
	# Here we combine the thresholded image (frameThresh) and the edge-detected image (frameEdge).
    frameFinal = cv2.add(frameThresh, frameEdge)
	# Then we convert the final image (frameFinal) from grayscale to RGB format using cv2.cvtColor().
    # This ensures that the final image has three channels (RGB) instead of one (grayscale).
    frameFinal = cv2.cvtColor(frameFinal, cv2.COLOR_GRAY2RGB)
    frameFinalDuplicate = cv2.cvtColor(frameFinal, cv2.COLOR_RGB2BGR)

def Histogram():
	# This function calculates the distribution of pixel intensities along the horizontal axis of the frame.
	# It focuses on our trapezoid region of interest within the frame to compute separate histograms.
	# These histograms help identify the presence and position of lane lines within the frame.

    global frameFinalDuplicate, histogramLane
	# Here we initialize an array 'histogramLane' with zeros to store the histogram data.
    # The array has a length of 400, corresponding to the width of the frame.
    histogramLane = np.zeros(400, dtype=int)

    # Loop through the width of the frame (400 pixels).
    for i in range(400):
		# Within each iteration, extract a small region of interest (ROI) from 'frameFinalDuplicate'.
        # The ROI corresponds to a vertical strip of the frame, from rows 140 to 240 and the current column 'i'.
        ROILane = frameFinalDuplicate[140:240, i:i+1]
		# Compute the sum of pixel values in the ROI and normalize it by dividing by 255.
        # This operation effectively counts the number of white pixels (representing lane lines) in the ROI.
        histogramLane[i] = int(np.sum(ROILane) / 255)


def LaneFinder():
	# Here, the previously calculated histograms are analyzed to determine the positions of lane boundaries.
	# By finding peaks in the histograms, the function identifies where the lines are most prominent.
	# These peak positions represent the left and right boundaries of the lane.
	# Lines are drawn on the frame to visualize these boundary positions.

    global histogramLane, LeftLanePos, RightLanePos, frameFinal
	# We find the position of the peak in the left lane histogram.
    LeftLanePos = np.argmax(histogramLane[:150])
	# We find the position of the peak in the right lane histogram.
    RightLanePos = np.argmax(histogramLane[250:]) + 250

    # We draw vertical lines on 'frameFinal' to visualize the left and right lane boundaries.
    # These lines are drawn at the positions determined by 'LeftLanePos' and 'RightLanePos'.
    cv2.line(frameFinal, (LeftLanePos, 0), (LeftLanePos, 240), (0, 255, 0), 2)
    cv2.line(frameFinal, (RightLanePos, 0), (RightLanePos, 240), (0, 255, 0), 2)


def LaneCenter():
	# This function computes the center of the lane based on the positions of the left and right lane boundaries.
	# It establishes a reference point at the center of the frame for comparison.
	# By calculating the midpoint between the left and right boundaries, the function determines the lane's center.
	# Lines representing the lane center and the frame center are drawn on the frame to provide visual feedback.
	# Additionally, the function calculates the deviation of the lane center from the frame center, which is essential for lane tracking and steering.
    global LeftLanePos, RightLanePos, laneCenter, frameCenter, Result, frameFinal
	# Compute the center of the lane by finding the midpoint between the left and right lane boundaries.
    laneCenter = (RightLanePos - LeftLanePos) // 2 + LeftLanePos
	# We set the frame center as a reference point for comparison that has to be adjusted based on the camera position.
    frameCenter = 192

    # We draw lines on 'frameFinal' to visualize the lane center and the frame center.
    cv2.line(frameFinal, (laneCenter, 0), (laneCenter, 240), (0, 255, 0), 3)
    cv2.line(frameFinal, (frameCenter, 0), (frameCenter, 240), (255, 0, 0), 3)
	# Here we compute the deviation of the lane center from the frame center.
    Result = laneCenter - frameCenter
    


def detect_stop_sign(frame):
    """ 
    This function detects stop signs in the input frame. It returns the frame with detected stop signs highlighted and the boolean variable 
    stop_detected indicating whether a stop sign was detected.
    """
    # Initialize variables
    stop_detected = False
    red_lower = (0, 50, 50)
    red_upper = (10, 255, 255)
	# This resizes the frame to the size (400x240) for processing
    resized_frame = cv2.resize(frame, (400, 240))
	# We apply Gaussian blur to the frame to reduce noise
    blurred_frame = cv2.GaussianBlur(resized_frame, (5, 5), 0)
	# This converts the blurred frame to HSV color space
    hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
	# We perform color segmentation
    mask = cv2.inRange(hsv_frame, red_lower, red_upper)
    # We apply morphological filters to the current frame: erosion and dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # Erosion
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    # Dilation
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
    # We perform shape segmentation
    edges = cv2.Canny(dilated_mask, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# We iterate through detected contours to identify potential stop sign
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        area = cv2.contourArea(contour)
		# We check if the contour resembles a stop sign shape (assuming octagonal)
        if len(approx) == 8:
			# Here we draw a bounding box around the stop sign
            if perimeter > 100 and area > 500:
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(resized_frame, 'STOP', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Set stop_detected flag to True indicating stop sign detection
                stop_detected = True
    return resized_frame, stop_detected # Return the resized frame with detected stop signs and stop_detected flag

    
def process_frame_stop(frame):
    # Start Stop signals detection
    processed_frame, stop_detected = detect_stop_sign(frame.copy())
    return processed_frame, stop_detected     

def main():
	global frame, ss
	import RPi.GPIO as GPIO
	from time import sleep
	last_stop_time = -5
	GPIO.setwarnings(False)
	
	# Pins for right Motor
	in1 = 3
	in2 = 4
	en_a = 2
	# Pins for left Motor
	in3 = 17
	in4 = 27
	en_b = 22
	
	# Setup GPIO
	GPIO.setmode(GPIO.BCM)
	GPIO.setup(in1, GPIO.OUT)
	GPIO.setup(in2, GPIO.OUT)
	GPIO.setup(en_a, GPIO.OUT)
	
	GPIO.setup(in3, GPIO.OUT)
	GPIO.setup(in4, GPIO.OUT)
	GPIO.setup(en_b, GPIO.OUT)
	
	# Setup PWM (Pulse Width Modulation) for controlling motor speed
    # Create PWM objects for the enable pins of the motors
    # This code sets up Pulse Width Modulation (PWM) to control the speed of the motors. PWM is a technique for controlling analog devices 
    # using digital signals. Here, PWM is used to control the speed of the motors connected to the Raspberry Pi GPIO pins. The GPIO.PWM() function 
    # creates PWM objects for the enable pins (en_a and en_b) of the motors. The PWM objects are then started with an initial duty cycle of 75%, 
    # which determines the speed of the motors. A duty cycle of 75% means the motors will run at 75% of their maximum speed.
	q = GPIO.PWM(en_a, 50) # PWM object for the right motor
	p = GPIO.PWM(en_b, 50) # PWM object for the left motor
	p.start(75)
	q.start(75)
	
	# Set initial motor directions
	GPIO.output(in1, GPIO.LOW)
	GPIO.output(in2, GPIO.LOW)
	GPIO.output(in4, GPIO.LOW)
	GPIO.output(in3, GPIO.LOW)
	
	# Wait for user input to start
	while True:
		user_input = input()
		if user_input == "start":
			break

	# Main loop for capturing frames and processing	
	while True:
		# Capture frame from camera
		Capture()
		# Detect stop sign in the frame
		stop_frame, stop_detected = process_frame_stop(frame)
		Perspective()
		Threshold()
		Histogram()
		LaneFinder()
		LaneCenter()
		
		# Perform motor control based on lane detection result
		if stop_detected == True and time.time() - last_stop_time >= 5:
			# Stop the car
			GPIO.output(in1, GPIO.LOW)
			GPIO.output(in2, GPIO.LOW)
			GPIO.output(in4, GPIO.LOW)
			GPIO.output(in3, GPIO.LOW)
			p.ChangeDutyCycle(75)
			q.ChangeDutyCycle(75)
			print("Stop!")
			sleep(3)
			last_stop_time = time.time()
		
		# Motor control based on lane detection result
		if Result == 0:
			# Move forward
			GPIO.output(in1, GPIO.HIGH)
			GPIO.output(in2, GPIO.LOW)
			GPIO.output(in4, GPIO.HIGH)
			GPIO.output(in3, GPIO.LOW)
			p.ChangeDutyCycle(75)
			q.ChangeDutyCycle(75)
			print("Forward")
		elif 0 < Result <7:
			# Turn right slightly
			GPIO.output(in1, GPIO.HIGH)
			GPIO.output(in2, GPIO.LOW)
			GPIO.output(in4, GPIO.HIGH)
			GPIO.output(in3, GPIO.LOW)
			p.ChangeDutyCycle(75)
			q.ChangeDutyCycle(50)
			print('Right1')
		elif 7 <= Result <= 15:
			# Turn right moderately
			GPIO.output(in1, GPIO.HIGH)
			GPIO.output(in2, GPIO.LOW)
			GPIO.output(in4, GPIO.HIGH)
			GPIO.output(in3, GPIO.LOW)
			p.ChangeDutyCycle(75)
			q.ChangeDutyCycle(25)
			print('Right2')
		elif Result > 15:
			# Turn right sharply
			GPIO.output(in1, GPIO.LOW)
			GPIO.output(in2, GPIO.LOW)
			GPIO.output(in4, GPIO.HIGH)
			GPIO.output(in3, GPIO.LOW)
			p.ChangeDutyCycle(75)
			q.ChangeDutyCycle(75)
			print('Right3')
		elif -7 < Result < 0:
			# Turn left slightly
			GPIO.output(in1, GPIO.HIGH)
			GPIO.output(in2, GPIO.LOW)
			GPIO.output(in4, GPIO.HIGH)
			GPIO.output(in3, GPIO.LOW)
			p.ChangeDutyCycle(50)
			q.ChangeDutyCycle(75)
			print('Left1')
		elif -15 <= Result <= -7:
			# Turn left moderately
			GPIO.output(in1, GPIO.HIGH)
			GPIO.output(in2, GPIO.LOW)
			GPIO.output(in4, GPIO.HIGH)
			GPIO.output(in3, GPIO.LOW)
			p.ChangeDutyCycle(25)
			q.ChangeDutyCycle(75)
			print('Left2')
		elif Result < -15:
			# Turn left sharply
			GPIO.output(in1, GPIO.HIGH)
			GPIO.output(in2, GPIO.LOW)
			GPIO.output(in4, GPIO.LOW)
			GPIO.output(in3, GPIO.LOW)
			p.ChangeDutyCycle(75)
			q.ChangeDutyCycle(75)
			print('Left3')
		
		# Display stop sign detection message if stop sign detected
		if stop_detected == True:
			cv2.putText(frame, "Stop detected", (1, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		else:
			ss = f"Result = {Result}"
			cv2.putText(frame, ss, (1, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		# We display original frame
		cv2.imshow("Original", frame)
		# We display perspective visual of the road from above
		cv2.imshow("Perspective", framePers)
		# We display final frame after all transformation
		cv2.imshow("Final", frameFinal)
		# We display frame with the stop signal detected
		cv2.imshow('Processed Frame for stop', stop_frame)

		# Check for 'q' key press to exit
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.destroyAllWindows()
	
	
if __name__ == "__main__":
	main()


