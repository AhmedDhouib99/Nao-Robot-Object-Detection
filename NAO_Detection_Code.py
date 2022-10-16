#import libreries
import os # Importation du Operating System
import numpy as np # Importation du numpy
import time # Importation du time
import cv2 # Importation du OpenCV
from naoqi import ALProxy
import sys
# Change the IpAddress value to your address IP
IpAddress = "192.168.16.104"
# Change the PORT value to your robot port value
PORT = 9559

# Creates a proxy on the Al-motion module
motion = ALProxy("ALMotion", IpAddress, PORT)
# get NAOqi module proxy
videoDevice = ALProxy('ALVideoDevice', IpAddress , PORT)
# Creates a proxy on the text-to-speech module
tts = ALProxy("ALTextToSpeech" , IpAddress , PORT)

motion.setStiffnesses("Head", 0.0)

# define paths and constants
LABELS_FILE='coco.names'
CONFIG_FILE='yolov3.cfg'
WEIGHTS_FILE='yolov3.weights'
CONFIDENCE_THRESHOLD=0.3


data=[]


# Creates a proxy on the Speech Recognition module
asr = ALProxy("ALSpeechRecognition", IpAddress, 9559)

asr.pause(True)

# define the language to use to English
asr.setLanguage("English")

# define the list of words to recognize
vocabulary = ["yes", "please", "what do you see "]

# Set the list
asr.setVocabulary(vocabulary, False)



#extend the objects name from the coconames and save them in list variable
LABELS = open(LABELS_FILE).read().strip().split("\n")

# define a color randomly
np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

#build the model
net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

#define the detection method

def detection (image ,layerOutputs ) :
	(H, W) = image.shape[:2]
	boxes = []
	confidences = []
	classIDs = []
	objects=[]
	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > CONFIDENCE_THRESHOLD:
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,CONFIDENCE_THRESHOLD)
	print(idxs)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			color = [int(c) for c in COLORS[classIDs[i]]]

			# delimit the boundaries of the object in rectangle shape and add the label
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)


			objects.append(LABELS[classIDs[i]])

	# return the list of objects detected
	return objects

#****************************************** acces to Robot camera **************************************************



# subscribe top camera
AL_kTopCamera = 0
AL_kQVGA = 1            # 320x240
AL_kBGRColorSpace = 13
captureDevice = videoDevice.subscribeCamera("test", AL_kTopCamera, AL_kQVGA, AL_kBGRColorSpace, 10)

# create image
width = 320
height = 240
image = np.zeros((height, width, 3), np.uint8)


while True :
	# *********/get image/*********
	result = videoDevice.getImageRemote(captureDevice)

	if result == None:
		print 'cannot capture.'
	elif result[6] == None:
		print 'no image data string.'
	else:
		# translate value to matric
		values = map(ord, list(result[6]))
		i = 0
		for y in range(0, height):
			for x in range(0, width):
				image.itemset((y, x, 0), values[i + 0])
				image.itemset((y, x, 1), values[i + 1])
				image.itemset((y, x, 2), values[i + 2])
				i += 3

	#**********************************


	# subscribe of the Speech Recognition module
	asr.subscribe(IpAddress)
	print("start speaking")
	# Creates a proxy on the Memory and subscribes to the event
	memProxy = ALProxy("ALMemory", IpAddress, 9559)
	memProxy.subscribeToEvent('WordRecognized', IpAddress, 'wordRecognized')

	asr.pause(False)
	time.sleep(2)

	asr.unsubscribe(IpAddress)
	print("stop speaking")

	data = memProxy.getData("WordRecognized")
	print("data: %s" % data)


	#***********************************

	#image = videoProxy.getImageRemote(subs)

	# Creation du Blob d'entrée du modele
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
	net.setInput(blob)
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	layerOutputs = net.forward(ln)
	# object detection part
	objects_detected=detection(image , layerOutputs)


	ch1= "a "
	ch = " ,a ".join(objects_detected[:-1])
	if len(objects_detected ) >1:
		ch+=" and  a {}".format(objects_detected[-1])
	ch=ch1+ch

	print(ch)

	if (ch != "a "):

		tts.say(" I am seeing {}".format(ch))
		time.sleep(2)
	else :
		pass

    # display the image
	cv2.imshow("NAO watch.png", image)

	#stop displaying the image if button clicked
	cv2.waitKey(1)