import numpy as np
import cv2
import time
from picamera2 import Picamera2


def region_selection(image):
    """
    This function identifies and extracts the region of interest from the captured frame after
    applying the Canny edge detector, which detects edges within the frame.
    """
    # Generate an array with the identical dimensions as the input image. 
    mask = np.zeros_like(image) 
    # Check if the provided frame has multiple channels
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255 # This is the color of the mask shape (white) which forms the area of interest at the bottom of the frame.
    # Here we construct a trapezoid-shaped area to focus only on the road in the frame.
    # We have created this polygon in accordance to how the camera was placed.
    bottom_left = [0, 240]
    top_left = [5, 195]
    bottom_right = [395, 240] 
    top_right = [400, 195]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    
    # Fill the shape with white color and create the final mask.
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # We use the AND operation between the input image and the mask to keep only the edges that are on the road.
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

def hough_transform(image):
    """
    The Hough Transform is a technique used in image processing to detect shapes, particularly lines or curves. It works by converting points in an image 
    to mathematical representations in parameter space, where shapes such as lines or curves are represented by mathematical equations. 
    This transformation allows for the detection of these shapes even when they are broken or distorted in the image.
    This function takes a grayscale image that comes out of the edge finder. This parameter should typically be a binary image, with edges highlighted using an edge detection algorithm like Canny.
    """
    # This is the distance resolution of the accumulator in pixels. It defines the granularity of the distance parameter of the Hough space.
    rho = 1
    # This parameter represents the angle resolution of the accumulator, measured in radians. It defines the granularity of the angle parameter of the Hough space.
    theta = np.pi/180
    # This is the minimum number of votes (intersections in Hough space) required for a detected line. Higher values will result in fewer detected lines being returned.
    threshold = 20
    #  This parameter specifies the minimum length of a line that will be accepted. Lines shorter than this length will be rejected.
    minLineLength = 20
    # This parameter defines the maximum allowed gap between segments to treat them as parts of the same line. If the gap between two segments is greater than this value, they will be considered separate lines.
    maxLineGap =500
    # This function is used to detect lines in an image using the Probabilistic Hough Transform. It returns an array containing 
    # dimensions of straight lines appearing in the input image.
    return cv2.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold, minLineLength = minLineLength, maxLineGap = maxLineGap)

def average_slope_intercept(lines):
    """
    This function calculates the average slope and intercept for the left and right lines in an image. This function takes in  input the output from 
    the Hough Transform, representing detected lines in the image.
    """
    # Lists to store the slopes, intercepts, and lengths of the left and right lines
    left_lines = [] # (slope, intercept)
    left_weights = [] # (length,)
    right_lines = [] # (slope, intercept)
    right_weights = [] # (length,)
    # Check if there are any detected lines
    if lines is not None:
        # Iterate through each detected line
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue
                # This computes the slope of the line detected
                slope = (y2 - y1) / (x2 - x1)
                # This computes the intercept of the line detected
                intercept = y1 - (slope * x1)
                # This computes the length of the line detected using Euclidean distance
                length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
                # Here we classify the line detected as left lane or right lane based on its slope
                if slope < 0: # Left lane has negative slope, right lane has positive slope
                    left_lines.append((slope, intercept)) # Store slope and intercept
                    left_weights.append((length)) # Store length
                else:
                    right_lines.append((slope, intercept)) # Store slope and intercept
                    right_weights.append((length)) # Store length
    # Here we compute the weighted average of slopes and intercepts for left and right lines
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    """
    This function serves the purpose of converting the mathematical representation of a line, defined by its slope and intercept, into a 
    practical representation in terms of pixel points on an image. In simpler terms, given the slope (which indicates the angle of the line) and 
    intercept (the point where the line crosses the y-axis) of a line, along with the y-coordinates of its starting (y1) and ending points (y2), this function 
    calculates the corresponding x-coordinates on the image. These x and y coordinates are then returned as a tuple, representing the starting and ending 
    points of the line in pixel coordinates.
    """
    # If the line is None, return None
    if line is None:
        return None
    # Then we extract slope and intercept from the line tuple
    slope, intercept = line
    if slope == 0:
        slope = 1
    # In this part we compute x-coordinates of the line's starting and ending points
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    # Return a tuple containing pixel points representing the line
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    """
    This function is designed to synthesize complete lines based on the pixel points derived from the output lines of the Hough Transform. 
    It returns a tuple that contains the coordinates of the left lane line and a tuple that contains the coordinates of the right lane line.
    """
    # We compute the slope and intercept of left and right lines
    left_lane, right_lane = average_slope_intercept(lines)
    # We calculate the y-coordinates for the starting and ending points of the lines
    y1 = image.shape[0] # Bottom of the image
    y2 = y1 * 0.6 # y2 is 60% of the height of the image
    # Convert slope and intercept into pixel points for left and right lines
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line # Return the left and right lane lines

def draw_lane_lines(image, lines, color=[255, 255, 0], thickness=12):
    """
    This function draws the lane lines onto the input frame.
    """
    # Create an array with all zeros with the same dimensions as input image
    line_image = np.zeros_like(image)
    # Iterate over each line detected by the Hough Transform
    for line in lines:
        # Check if the line is not None
        if line is not None:
            # Draw the line on the image
            cv2.line(line_image, *line, color, thickness)
    # Combine the original image with the image containing the drawn lines using weighted addition.
    # The drawn lines appear on top of the original image.
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0), lines


def draw_centers_and_compute_result(frame, lines):
    """
    This function calculates a central line between the left and right lines, and a line placed at the center of the frame. If the 
    central line of the frame (the blue one) is to the right of the green line representing the center of the lane, it means the car should turn left 
    because the road curves left. So, the value of the result will contain a strictly negative value indicating that the car should turn left. If instead 
    the blue line is to the left of the green one, it means there is a curve to the right, and therefore the result will take values strictly greater 
    than 0. If the lane is straight relative to the car, the blue and green lines will tend to overlap, and the result will be 0.
    """
    # Extract the coordinates of the lines
    left_line, right_line = lines
    # Here we check if any of the lines have missing coordinates
    if left_line is None:
        # If the left line is missing, return the original frame with a result equals to -31 (Left3)
        return frame, -31
    elif right_line is None:
        # If the right line is missing, return the original frame with a result equals to 31 (Right3)
        return frame, 31
    
    # Compute the maximum height of the two lines
    max_height = min(left_line[1][1], right_line[1][1])
    
    # Compute the midpoint coordinates of the two lines
    center_left_x = (left_line[0][0] + left_line[1][0]) // 2
    
    center_right_x = (right_line[0][0] + right_line[1][0]) // 2
    
    # Compute the midpoint between the two lines
    center_x = (center_left_x + center_right_x) // 2
    
    # Draw a green line at the midpoint between the two lines up to the maximum height
    cv2.line(frame, (center_x, frame.shape[1]), (center_x, max_height), (0, 255, 0), 2)
    
    # Cmpute the center of the frame
    frame_center_x = frame.shape[1] // 2
    
    # Draw a blue line at the center of the frame up to the maximum height
    cv2.line(frame, (frame_center_x, frame.shape[1]), (frame_center_x, max_height),  (255, 0, 0), 2)
    # Calculate the result as the difference between the detected center and the frame center
    result = center_x - frame_center_x

    # Prepare text displaying the result
    text = f"Result: {result}"

    # Draw the text on the frame
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Return the frame with the result code
    return frame, result


def frame_processor(image):
    """
    This function starts the process of detecting lane lines in the input frame.
    It calls various functions responsible for different stages of the lane detection process.
    """
    # Convert the input frame from BGR to RGB color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert from RGB image to Grayscale. The RGB image is converted to grayscale using cv2.cvtColor() again. Grayscale images contain 
    # only intensity information, which simplifies further processing.
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Gaussian blur is applied to the grayscale image. Gaussian blur smoothens the image by reducing noise and details. It's essential 
    # for removing high-frequency noise, which can lead to false edge detections. A Gaussian kernel of size 5x5 is used here.
    kernel_size = 5
    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
    # First threshold for the hysteresis procedure.
    low_t = 50
    # Second threshold for the hysteresis procedure. 
    high_t = 400
    # Canny edge detection uses the concept of gradients to identify edges. It calculates the gradient magnitude 
    # and orientation at each pixel in the image. The gradient magnitude represents the rate of change of intensity, while the gradient 
    # orientation indicates the direction of the change. Canny edge detection applies two thresholds to identify potential edge pixels: 
    # a low threshold (low_t) and a high threshold (high_t). Pixels with gradient magnitudes above the high threshold are considered strong edge pixels, 
    # while those below the low threshold are considered weak edge pixels. To obtain continuous edges and suppress noise, Canny edge detection performs 
    # edge tracking by hysteresis. Weak edge pixels are connected to strong edge pixels if they are adjacent. This helps in linking 
    # discontinuous edge segments into continuous edges and suppressing noise.
    edges = cv2.Canny(blur, low_t, high_t)
    # A region of interest (ROI) mask is applied to the Canny edges image. This mask focuses only on the region of interest, 
    # which is, in our case, the road. This step helps in filtering out unwanted edges that are not part of the road.
    region = region_selection(edges)
    # The Hough Transform is applied to the region of interest to detect straight lines, which represent the lane lines on the road. The Probabilistic 
    # Hough Transform is used here, which is more efficient than the standard Hough Transform.
    hough = hough_transform(region)
    # Detected lane lines are drawn on the original RGB image 
    lane_detected_image, lines = draw_lane_lines(image, lane_lines(image, hough))
    # The distance of the vehicle from the center of the lane is computed based on the detected lane lines. The center of the lane is 
    # determined by averaging the positions of the left and right lane lines. The result is the difference between the center of the image and 
    # the center of the lane.
    frame_with_centers, Result = draw_centers_and_compute_result(lane_detected_image, lines)
    return frame_with_centers, Result # The final output of the method consists of the original image with detected lane lines drawn on it and the computed result
    
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
    hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_RGB2HSV)
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
		# We check if the contour has an octagonal shape
        if len(approx) == 8:
			# Here we draw a bounding box around the stop sign
            if perimeter > 100 and area > 500:
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(resized_frame, 'STOP', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Set stop_detected flag to True indicating stop sign detection
                stop_detected = True
    return cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB), stop_detected # Return the resized frame with detected stop signs and stop_detected flag


def process_frame_stop(frame):
    """
    Start Stop signals detection
    """
    processed_frame, stop_detected = detect_stop_sign(frame.copy())
    return processed_frame, stop_detected


def capture_frames():
     
	from time import sleep
	import RPi.GPIO as GPIO
     
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
	
    # Setup PWM (Pulse Width Modulation) for controlling motor speed.
    # Create PWM objects for the enable pins of the motors.
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
	
    # Initialize Picamera2 camera
	camera = Picamera2()
	camera.configure(camera.create_preview_configuration(main={"size": (400, 240)}))
	camera.start()
    # Main loop for capturing frames and processing
	while True:
		# Capture frame from camera
		frame = camera.capture_array()
		frame = cv2.resize(frame, (400, 240))
        # Detect stop sign in the frame
		stop_frame, stop_detected = process_frame_stop(frame)
        # Process frame for lane detection
		lane_detected_frame, Result = frame_processor(frame)
		
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
		elif 0 < Result <15:
            # Turn right slightly
			GPIO.output(in1, GPIO.HIGH)
			GPIO.output(in2, GPIO.LOW)
			GPIO.output(in4, GPIO.HIGH)
			GPIO.output(in3, GPIO.LOW)
			p.ChangeDutyCycle(75)
			q.ChangeDutyCycle(50)
			print('Right1')
		elif 15 <= Result <= 30:
            # Turn right moderately
			GPIO.output(in1, GPIO.HIGH)
			GPIO.output(in2, GPIO.LOW)
			GPIO.output(in4, GPIO.HIGH)
			GPIO.output(in3, GPIO.LOW)
			p.ChangeDutyCycle(75)
			q.ChangeDutyCycle(25)
			print('Right2')
		elif Result > 30:
            # Turn right sharply
			GPIO.output(in1, GPIO.LOW)
			GPIO.output(in2, GPIO.LOW)
			GPIO.output(in4, GPIO.HIGH)
			GPIO.output(in3, GPIO.LOW)
			p.ChangeDutyCycle(75)
			q.ChangeDutyCycle(75)
			print('Right3')
		elif -15 < Result < 0:
            # Turn left slightly
			GPIO.output(in1, GPIO.HIGH)
			GPIO.output(in2, GPIO.LOW)
			GPIO.output(in4, GPIO.HIGH)
			GPIO.output(in3, GPIO.LOW)
			p.ChangeDutyCycle(50)
			q.ChangeDutyCycle(75)
			print('Left1')
		elif -30 <= Result <= -15:
            # Turn left moderately
			GPIO.output(in1, GPIO.HIGH)
			GPIO.output(in2, GPIO.LOW)
			GPIO.output(in4, GPIO.HIGH)
			GPIO.output(in3, GPIO.LOW)
			p.ChangeDutyCycle(25)
			q.ChangeDutyCycle(75)
			print('Left2')
		elif Result < -30:
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
		
        # Display frames with lane detection and stop sign detection
		cv2.imshow('Lane Detection', lane_detected_frame)
		cv2.imshow('Processed Frame for stop', stop_frame)
          
		# Check for 'q' key press to exit
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_frames()
