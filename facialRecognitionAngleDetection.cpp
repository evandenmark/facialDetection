
#define _USE_MATH_DEFINES
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <algorithm>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);
vector<Rect> detectNoses(Mat faceROI, Rect face);
vector<Rect> detectMouths(Mat faceROI, Rect face);
vector<Rect> detectFaces(Mat frame_gray);
vector<float> displayNose(Rect nose, Mat frame, Rect face);
vector<float> displayMouth(Rect mouth, Mat frame, Rect face);
void displayFace(Mat frame, Rect face);
Rect determineCorrectNose(vector<Rect> noses, Rect face);
Rect determineCorrectMouth(vector<Rect> mouths, Rect face);
vector<float> getVariance(vector<Rect> attributes);
vector<float> getAvgXYCenter(vector<Rect> attributes);
Rect findSingleBestAttribute(vector<Rect> attributes, float avgX, float avgY, float stndDeviationX, float stndDeviationY, float STD_DEV_CONSTANT);
Rect getAvgAttribute(vector<Rect> attributes);
vector<float> determineAngle(vector<float> configuration, vector<float> nosePositions, vector<float> mouthPositions);
float determineDistance(vector<float> configuration, Rect face);
vector<float> parseConfigFile(String fileName);

/** Global variables */

string window_name = "Capture - Face detection";
RNG rng(12345);
float MOUTH_STANDARD_DEVIATIONS = 2.0;
float NOSE_STANDARD_DEVIATIONS = 2.0;
String face_cascade_name = "C:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
String mouth_cascade_name = "C:/opencv/opencv/sources/data/haarcascades/mouth.xml";
String nose_cascade_name = "C:/opencv/opencv/sources/data/haarcascades/nose.xml";
String CONFIG_FILE = "C:/Users/Evan/Desktop/configuration.txt";
CascadeClassifier face_cascade, mouth_cascade, nose_cascade;


/*
Runs the program. Takes the camera input and determines
1. If there is a face in the shot
2. If so, finds
a. the face's distance from the camera
b. yaw rotation
c. pitch rotation
*/
int main(int argc, const char** argv)
{
	// Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade\n"); return -1; };
	if (!mouth_cascade.load(mouth_cascade_name)) { printf("--(!)Error loading mouth cascade\n"); return -1; };
	if (!nose_cascade.load(nose_cascade_name)) { printf("--(!)Error loading nose cascade\n"); return -1; }

	//Open the video stream (0 is the webcam)
	VideoCapture cap(0);

	if (!cap.isOpened()) {
		return -1;
	}
	//go through 100 frames
	for (int q = 0; q<100; q++) {
		Mat frame;
		cap >> frame;
		if (!frame.empty())
		{
			detectAndDisplay(frame);
		}
		else
		{
			printf(" --(!) No captured frame -- Break!"); break;
		}
		int c = waitKey(1);
		if ((char)c == 'c') { break; }
	}
	return 0;
}

//Finds faces, mouths, and noses
//Performs the bulk of computation
void detectAndDisplay(Mat frame)
{
	//initialization
	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	if (faces.size() > 1) {
		cout << "Hey! More than 1 face was detected in the scene." << endl;
	}
	//Note: only goes through 1 face here (sometimes nonexistent faces are detected)
	for (size_t i = 0; i < faces.size(); i++)
	{
		vector<float> nosePositions, mouthPositions;

		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

		Mat faceROI = frame_gray(faces[i]);

		//detect, filter, and display noses
		vector<Rect> noses = detectNoses(faceROI, faces[i]);
		Rect bestNose = determineCorrectNose(noses, faces[i]);
		nosePositions = displayNose(bestNose, frame, faces[i]); //normalized to the face ( betweeen 0 and 1)

		//detect, filter, and display mouths
		vector<Rect> mouths = detectMouths(faceROI, faces[i]);
		Rect bestMouth = determineCorrectMouth(mouths, faces[i]);
		mouthPositions = displayMouth(bestMouth, frame, faces[i]); //normalized to the face

		/*
		vector<float> configuration{	0.45f, 0.0f,0.4f,0.0f,
										0.32f, 0.52f,0.34f,0.7f,
										0.24f, 0.0f,0.29f,0.0f,
										0.0f,0.57f,0.0f,0.8f,
										0.0f,0.3f,0.0f,0.6f,
										200.0f };
		
		*/

		//NOTE: CONFIG FILE IS BEING PARSED EVERY FRAME
		vector<float> configuration = parseConfigFile(CONFIG_FILE);
		
		//calculate yaw and pitch rotations
		vector<float> angles = determineAngle(configuration, nosePositions, mouthPositions);

		//determine how far away the face is to the camera
		float distance = determineDistance(configuration, faces[i]);
		
		cout << "Dist:  " << distance << endl;
		cout << "Yaw:   " << angles[0] << endl;
		cout << "Pitch: " << angles[1] << endl;
	}
	imshow(window_name, frame);
}


vector<float> parseConfigFile(String fileName) {
	ifstream myFile;
	myFile.open(fileName);
	vector<float> configuration;
	std::string line;
	while (getline(myFile, line)) {
		std::istringstream iss(line);
		float noseX, noseY, mouthX, mouthY, width;
		if (iss >> width) {
			configuration.push_back(width);
		}
		else { break; }

		if (iss >> noseY >> mouthX >> mouthY) {
			configuration.push_back(noseY);
			configuration.push_back(mouthX);
			configuration.push_back(mouthY);
		}
		else { break; }
	}
	return configuration;

}

//Using initial configurations, determines how far away the face is from the camera
//The observed size of the face falls off as 1/distance ratio
float determineDistance(vector<float> configuration, Rect face) {

	float initalDistance = 0.6; //meters
	float initalHeadWidth = configuration[20]; //pixels
	float currentHeadWidth = face.width; //pixels

	return initalDistance*initalHeadWidth / currentHeadWidth; //meters
}

vector<float> determineAngle(vector<float> configuration, vector<float> nosePositions, vector<float> mouthPositions) {
	// config {leftNoseX, leftNoseY, leftMouthX, leftMouthY,
	//		    midNoseX, midNoseY, midMouthX, midMouthY
	//			rightNoseX, rightNoseY, rightMouthX, rightMouthY,
	//			downNoseX, downNoseY, downMouthX, downMouthY,
	//			upNoseX, upNoseY, upMouthX, upMouthY,
	//			faceWidth}

	float leftNoseXConfig = configuration[0]; float leftMouthXConfig = configuration[2];
	float midNoseXConfig = configuration[4]; float midMouthXConfig = configuration[6];
	float rightNoseXConfig = configuration[8]; float rightMouthXConfig = configuration[10];
	float currentNoseX = nosePositions[0]; float currentMouthX = mouthPositions[0];

	float downNoseYConfig = configuration[13]; float downMouthYConfig = configuration[15];
	float midNoseYConfig = configuration[5]; float midMouthYConfig = configuration[7];
	float upNoseYConfig = configuration[17]; float upMouthYConfig = configuration[19];
	float currentNoseY = nosePositions[1]; float currentMouthY = mouthPositions[1];

	float yawAngleFromNose, yawAngleFromMouth, distanceBetween, yawAngle, pitchAngle, pitchAngleFromNose, pitchAngleFromMouth;
	float midAngleValue = 0.0;
	float leftAngleValue = 50.0;
	float rightAngleValue = -40.0;
	float downAngleValue = -30.0;
	float upAngleValue = 50.0;

	//determine yaw (horizontal) angle
	distanceBetween = abs(leftNoseXConfig - midNoseXConfig);
	yawAngleFromNose = ((1 - ((currentNoseX - midNoseXConfig) / distanceBetween))*midAngleValue) + ((currentNoseX - midNoseXConfig) / distanceBetween)*leftAngleValue;

	distanceBetween = abs(leftMouthXConfig - midMouthXConfig);
	yawAngleFromMouth = ((1 - ((currentMouthX - midMouthXConfig) / distanceBetween))*midAngleValue) + ((currentMouthX - midMouthXConfig) / distanceBetween)*leftAngleValue;
	

	//determine pitch (vertical) angle
	if (currentNoseY < midNoseYConfig) {
		distanceBetween = abs(upNoseYConfig - midNoseYConfig);
		pitchAngleFromNose = ((1 - (abs(currentNoseY - midNoseYConfig) / distanceBetween))*midAngleValue) + (abs(currentNoseY - midNoseYConfig) / distanceBetween)*upAngleValue;
	}
	else {
		distanceBetween = abs(downNoseYConfig - midNoseYConfig);
		pitchAngleFromNose = ((1 - (abs(currentNoseY - midNoseYConfig) / distanceBetween))*midAngleValue) + (abs(currentNoseY - midNoseYConfig) / distanceBetween)*downAngleValue;
	}
	distanceBetween = abs(downMouthYConfig - midMouthYConfig);
	pitchAngleFromMouth = ((1 - ((currentMouthY - midMouthYConfig) / distanceBetween))*midAngleValue) + ((currentMouthY - midMouthYConfig) / distanceBetween)*downAngleValue;
	
	if (currentNoseX == 0 && currentNoseY == 0) {
		//no nose was detected
		yawAngle = yawAngleFromMouth;
		pitchAngle = pitchAngleFromMouth;
	}
	else {
		yawAngle = yawAngleFromNose;
		pitchAngle = pitchAngleFromNose;
	}
	return{ yawAngle, pitchAngle };

}

//for a vector of attributes (face, nose, mouth, etc..), returns a Rect representing
//the average of position and size of the attributes Rect(x,y,width,height)
Rect getAvgAttribute(vector<Rect> attributes) {
	float totalX = 0.0;
	float totalY = 0.0;
	float totalWidth = 0.0;
	float totalHeight = 0.0;
	int s = attributes.size();
	for (int i = 0; i < attributes.size(); i++) {
		totalX += attributes[i].x;
		totalY += attributes[i].y;
		totalWidth += attributes[i].width;
		totalHeight += attributes[i].height;
	}
	return Rect(totalX / float(s), totalY / float(s), totalWidth / float(s), totalHeight / float(s));
}

//Given a vector of noses, finds a single nose that best represents the group
Rect determineCorrectNose(vector<Rect> noses, Rect face) {

	if (noses.size() == 0) {
		//edge case: if there is no nose found, return an empty Rect
		return Rect();
	}

	//get the spatial variance of the mouth locations
	float varianceX = getVariance(noses)[0];
	float varianceY = getVariance(noses)[1];
	float stdDevX = sqrt(varianceX);
	float stdDevY = sqrt(varianceY);
	float avgX = getAvgXYCenter(noses)[0];
	float avgY = getAvgXYCenter(noses)[1];

	//chop off outliers
	return findSingleBestAttribute(noses, avgX, avgY, stdDevX, stdDevY, NOSE_STANDARD_DEVIATIONS);
}


Rect determineCorrectMouth(vector<Rect> mouths, Rect face) {

	// simple check if mouths are in bottom half of face
	float faceCenterX = face.width / 2.0;
	float faceCenterY = face.height / 2.0;
	vector<Rect> newMouths;
	for (int i = 0; i < mouths.size(); i++) {
		if (mouths[i].y > faceCenterY) {
			newMouths.push_back(mouths[i]);
		}
	}

	if (newMouths.size() == 0) {
		//edge case: if no mouths are in the bottom half of the face, return empty Rect
		return Rect();
	}

	//get the spatial variance of the mouth locations
	float varianceX = getVariance(newMouths)[0];
	float varianceY = getVariance(newMouths)[1];
	float stdDevX = sqrt(varianceX);
	float stdDevY = sqrt(varianceY);
	float avgX = getAvgXYCenter(newMouths)[0];
	float avgY = getAvgXYCenter(newMouths)[1];

	//chop off outliers who are not within 2 std deviations
	return findSingleBestAttribute(newMouths, avgX, avgY, stdDevX, stdDevY, MOUTH_STANDARD_DEVIATIONS);
}

//detect all of the noses within the face
vector<Rect> detectNoses(Mat faceROI, Rect face) {
	vector<Rect> noses;
	nose_cascade.detectMultiScale(faceROI, noses, 1.1, 0, 0 | CASCADE_SCALE_IMAGE , Size(face.width / 5.5, face.width / 5.5), Size(face.width / 3.0, face.width / 3.0));
	return noses;
}

//detect all of the mouths within the face
vector<Rect> detectMouths(Mat faceROI, Rect face) {
	vector<Rect> mouths;
	mouth_cascade.detectMultiScale(faceROI, mouths, 1.1, 0, 0 | CASCADE_SCALE_IMAGE, Size(face.width / 4.0, face.height / 8.0), Size(face.width / 2.5, face.height / 5.0));
	//cout << "NUM MOUTHS: " << mouths.size() << endl;
	return mouths;
}

//puts an ellipse in the frame at the location of the nose
//returns the location of the nose with respect to the face
vector<float> displayNose(Rect nose, Mat frame, Rect face) {

	Point nCenter(face.x + nose.x + nose.width / 2, face.y + nose.y + nose.height / 2);
	ellipse(frame, nCenter, Size(nose.width / 4, nose.height / 4), 0, 0, 360, Scalar(255, 255, 0), 4, 8, 0);
	return{ nose.x / float(face.width) , nose.y / float(face.height) };
}

//puts an ellipse in the frame at the location of the mouth
//returns the location of the mouth with respect to the face
vector<float> displayMouth(Rect mouth, Mat frame, Rect face) {

	Point mCenter(face.x + mouth.x + mouth.width / 2, face.y + mouth.y + mouth.height / 2);
	ellipse(frame, mCenter, Size(mouth.width*0.75, mouth.height / 2), 0, 0, 360, Scalar(0, 255, 0), 4, 8, 0);
	return{ mouth.x / float(face.width) , mouth.y / float(face.height) };
}

//gets the variance of an attribute distribution
//returns vector  representing <xVariance, yVariance> 
vector<float> getVariance(vector<Rect> attributes) {

	//get spatial variance
	float avgX = getAvgXYCenter(attributes)[0];
	float avgY = getAvgXYCenter(attributes)[1];

	float sumTotalX = 0.0;
	float sumTotalY = 0.0;
	for (int j = 0; j < attributes.size(); j++) {
		float attributeCenterX = attributes[j].x + attributes[j].width / 2.0;
		float attributeCenterY = attributes[j].y + attributes[j].height / 2.0;
		sumTotalX += pow((attributeCenterX - avgX), 2.0);
		sumTotalY += pow((attributeCenterY - avgY), 2.0);
	}
	float varianceX = sumTotalX / float(attributes.size());
	float varianceY = sumTotalY / float(attributes.size());
	vector<float> variances{ varianceX, varianceY };
	return variances;
}

//given a vector of attributes, gets the average x and y coordinates of the attributes
vector<float> getAvgXYCenter(vector<Rect> attributes) {
	float totalX = 0.0;
	float totalY = 0.0;
	float totalWeight = 0.0;
	float totalHeight = 0.0;
	for (int i = 0; i < attributes.size(); i++) {
		float attributeCenterX = attributes[i].x + attributes[i].width / 2.0;
		float attributeCenterY = attributes[i].y + attributes[i].height / 2.0;
		totalX += attributeCenterX;
		totalY += attributeCenterY;
	}
	float avgX = totalX / float(attributes.size());
	float avgY = totalY / float(attributes.size());
	vector<float> avg{ avgX, avgY };
	return avg;
}

//given a list of attributes, determines the most probable location and size 
//of the best attribute
Rect findSingleBestAttribute(vector<Rect> attributes, float avgX, float avgY, float stndDeviationX, float stndDeviationY, float STD_DEV_CONSTANT) {

	float totalWidth = 0.0;
	float totalHeight = 0.0;

	vector<Rect> newAttributes;
	for (int i = 0; i < attributes.size(); i++) {
		float attributesCenterX = attributes[i].x + attributes[i].width / 2.0;
		float attributesCenterY = attributes[i].y + attributes[i].height / 2.0;
		if (abs(attributesCenterX - avgX) <= stndDeviationX*STD_DEV_CONSTANT
			&& abs(attributesCenterY - avgY) <= stndDeviationY*STD_DEV_CONSTANT) {
			totalWidth += attributes[i].width;
			totalHeight += attributes[i].height;
			newAttributes.push_back(attributes[i]);
		}
	}
	float avgWidth = totalWidth / newAttributes.size();
	float avgHeight = totalHeight / newAttributes.size();
	float newAvgX = getAvgXYCenter(newAttributes)[0];
	float newAvgY = getAvgXYCenter(newAttributes)[1];
	//take avg center and dimensions
	return Rect(newAvgX - avgWidth / 2.0, newAvgY - avgHeight / 2.0, avgWidth, avgHeight);
}
