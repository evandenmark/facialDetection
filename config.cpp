
#define _USE_MATH_DEFINES
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "windows.h"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);
vector<Rect> detectNoses(Mat faceROI, Rect face);
vector<Rect> detectMouths(Mat faceROI, Rect face);
vector<Rect> detectProfiles(Mat frame_gray);
vector<Rect> detectFaces(Mat frame_gray);
vector<float> displayNose(Rect nose, Mat frame, Rect face);
vector<float> displayMouth(Rect mouth, Mat frame, Rect face);
void displayFace(Mat frame, Rect face);
void displayProfile(Mat frame, Rect profile);
Rect determineCorrectNose(vector<Rect> noses, Rect face);
Rect determineCorrectMouth(vector<Rect> mouths, Rect face);
Rect getBestFace(vector<Rect> faces, vector<Rect> profiles);
vector<float> getVariance(vector<Rect> attributes);
vector<float> getAvgXYCenter(vector<Rect> attributes);
Rect findSingleBestAttribute(vector<Rect> attributes, float avgX, float avgY, float stndDeviationX, float stndDeviationY, float STD_DEV_CONSTANT);
Rect getAvgAttribute(vector<Rect> attributes);
float determineAngle(vector<float> configuration, vector<float> nosePositions, vector<float> mouthPositions);
void spaceBarPressed();
void writeConfigFile(vector<float> left, vector<float> mid, vector<float> right, vector<float> down, vector<float> up, float headwidth);
float getAvg(vector<float> v);
vector<float> readPoints();
vector<float> detectNoseAndMouthPositions(Mat frame);
bool goodSample(vector<float> leftNoseX, vector<float> leftNoseY, vector<float> leftMouthX, vector<float> leftMouthY);
float detectHeadWidth(Mat frame);
float getHeadWidth();
int numZeros(vector<float> v);

/** Global variables */

string window_name = "Capture - Face detection";
RNG rng(12345);
float FACE_STANDARD_DEVIATIONS = 2.5;
float MOUTH_STANDARD_DEVIATIONS = 2.0;
float NOSE_STANDARD_DEVIATIONS = 2.0;
float EYES_STANDARD_DEVIATIONS = 2.0;
String face_cascade_name = "C:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "C:/opencv/opencv/sources/data/haarcascades/haarcascade_eye.xml";
String profile_face_cascade_name = "C:/opencv/opencv/sources/data/haarcascades/haarcascade_profileface.xml";
String mouth_cascade_name = "C:/opencv/opencv/sources/data/haarcascades/mouth.xml";
String nose_cascade_name = "C:/opencv/opencv/sources/data/haarcascades/nose.xml";
CascadeClassifier face_cascade, eyes_cascade, profile_cascade, mouth_cascade, nose_cascade;
vector<vector<float>> allFacialFeaturesPositions;


/** @function main */
int main(int argc, const char** argv)
{

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading eyes cascade\n"); return -1; };
	if (!profile_cascade.load(profile_face_cascade_name)) { printf("--(!)Error loading profile cascade\n"); return -1; }
	if (!mouth_cascade.load(mouth_cascade_name)) { printf("--(!)Error loading mouth cascade\n"); return -1; };
	if (!nose_cascade.load(nose_cascade_name)) { printf("--(!)Error loading nose cascade\n"); return -1; }

	cout << "Hi. Welcome to the Facial Configuration." << endl;
	
	cout << "Please turn your head to the left about 50 degrees (or until the neck starts to strain)." << endl;
	cout << "When you are ready, press the spacebar." << endl;
	spaceBarPressed();
	vector<float> leftPoints = readPoints();
	cout << "Great. Left turn config was a success. Now, look straight at the camera." << endl;
	cout << "When you are ready, press the spacebar." << endl;
	spaceBarPressed();
	vector<float> middlePoints = readPoints();
	float headWidth = getHeadWidth();
	cout << "Great. Middle config was a success. Now, turn your head to the right about 50 degrees (or until the neck strains)." << endl;
	cout << "When you are ready, press the spacebar." << endl;
	spaceBarPressed();
	vector<float> rightPoints = readPoints();
	cout << "Great. Right config was a success. Now, tilt downwards approximately 30 degrees." << endl;
	cout << "When you are ready, press the spacebar." << endl;
	spaceBarPressed();
	vector<float> downPoints = readPoints();
	cout << "Great. Down config was a success. Now, tilt you head up about 50 degrees." << endl;
	cout << "When you are ready, press the spacebar." << endl;
	spaceBarPressed();
	vector<float> upPoints = readPoints();
	cout << "Configuration complete." << endl;
	writeConfigFile(leftPoints, middlePoints, rightPoints, downPoints, upPoints, headWidth);

	return 0;
}

void spaceBarPressed() {
	bool pressed = false;
	while (!pressed) {
		if (GetAsyncKeyState(VK_SPACE) != 0) {
			pressed = true;
		}
	}
}

void writeConfigFile(vector<float> left, vector<float> mid, vector<float> right, vector<float> down, vector<float> up, float headwidth) {
	ofstream myFile;
	myFile.open("C:/Users/Evan/Desktop/configuration.txt");
	float leftNoseX = left[0]; float leftNoseY = left[1]; float leftMouthX = left[2];	float leftMouthY = left[3];
	float midNoseX = mid[0]; float midNoseY = mid[1]; float midMouthX = mid[2]; float midMouthY = mid[3];
	float rightNoseX = right[0]; float rightNoseY = right[1]; float rightMouthX = right[2]; float rightMouthY = right[3];
	float downNoseX = down[0]; float downNoseY = down[1]; float downMouthX = down[2]; float downMouthY = down[3];
	float upNoseX = up[0]; float upNoseY = up[1]; float upMouthX = up[2]; float upMouthY = up[3];
	
	myFile << leftNoseX << " " << leftNoseY << " " << leftMouthX << " " << leftMouthY << endl;
	myFile << midNoseX << " " << midNoseY << " " << midMouthX << " " << midMouthY << " " << endl;
	myFile << rightNoseX << " " << rightNoseY << " " << rightMouthX << " " << rightMouthY << endl;
	myFile << downNoseX << " " << downNoseY << " " << downMouthX << " " << downMouthY << endl;
	myFile << upNoseX << " " << upNoseY << " " << upMouthX << " " << upMouthY << endl;
	myFile << headwidth << endl;
	myFile.close();
}

vector<float> readPoints() {

	VideoCapture cap(0);

	if (!cap.isOpened()) {
		cout << "Can't open the camera!" << endl;
		return{};
	}
	vector<float> NoseX;
	vector<float> NoseY;
	vector<float> MouthX;
	vector<float> MouthY;
	NoseX.reserve(10);
	NoseY.reserve(10);
	MouthX.reserve(10);
	MouthY.reserve(10);
	for (int q = 0; q<10; q++) {
		Mat frame;
		cap >> frame;
		if (!frame.empty())
		{
			vector<float> framePoints = detectNoseAndMouthPositions(frame);
			
			NoseX.push_back(framePoints[0]);
			NoseY.push_back(framePoints[1]);
			MouthX.push_back(framePoints[2]);
			MouthY.push_back(framePoints[3]);
		}
		else
		{
			printf(" --(!) No captured frame -- Break!"); break;
		}
		int c = waitKey(10);
		if ((char)c == 'c') { break; }
	}
	if (goodSample(NoseX, NoseY, MouthX, MouthY)) {
		cout << getAvg(NoseX) << " " << getAvg(NoseY) << " " << getAvg(MouthX) << " " << getAvg(MouthY) << endl;
		return{ getAvg(NoseX), getAvg(NoseY) ,getAvg(MouthX) ,getAvg(MouthY) };
	}
	else {
		cout << "Sorry. Your readings didn't come in very clear. Let's try it again." << endl;
		cout << "Try rotating your head a little bit less." << endl << endl;
		cout << "When you are ready, press the spacebar." << endl;
		spaceBarPressed();
		cap.release();
		return readPoints();
	}
	cap.release();
}

bool goodSample(vector<float> NoseX, vector<float> NoseY, vector<float> MouthX, vector<float> MouthY) {
	
	if (numZeros(NoseX) > NoseX.size() / 2 || numZeros(NoseY) > NoseY.size() / 2 ||
		numZeros(MouthX) > MouthX.size() / 2 || numZeros(MouthY) > NoseY.size() / 2) {
		//over half of the samples did not find a nose or mouth 
		return false;
	}
	return true;
}

int numZeros(vector<float> v) {
	int total = 0;
	for (int i = 0; i < v.size(); i++) {
		if (v[i] == 0) {
			total += 1;
		}
	}
	return total;
}

float getAvg(vector<float> v) {
	float total = 0.0;
	float divisor = 0.0;
	for (int i = 0; i < v.size(); i++) {
		if (v[i] != 0) {
			divisor += 1;
		}
		total += v[i];
	}
	if (divisor == 0.0) {
		return 0;
	}
	return total / divisor;
}

float getHeadWidth() {
	VideoCapture cap(0);

	if (!cap.isOpened()) {
		cout << "Can't open the camera!" << endl;
		return{};
	}
	vector<float> hWidths;
	for (int q = 0; q<3; q++) {
		Mat frame;
		cap >> frame;
		//-- 3. Apply the classifier to the frame
		if (!frame.empty())
		{
			hWidths.push_back(detectHeadWidth(frame));
		}
		else
		{
			printf(" --(!) No captured frame -- Break!"); break;
		}
		int c = waitKey(10);
		if ((char)c == 'c') { break; }
	}
	return getAvg(hWidths);
}

float detectHeadWidth(Mat frame) {
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	return faces[0].width;
}

vector<float> detectNoseAndMouthPositions(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	if (faces.size() > 1) {
		cout << "Hey! More than 1 face was detected in the scene." << endl;
	}
	for (size_t i = 0; i < faces.size(); i++)
	{
		vector<float>  nosePositions, mouthPositions;
		nosePositions.reserve(2);
		mouthPositions.reserve(2);
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

		Mat faceROI = frame_gray(faces[i]);

		vector<Rect> noses = detectNoses(faceROI, faces[i]);
		Rect bestNose = determineCorrectNose(noses, faces[i]);
		nosePositions = displayNose(bestNose, frame, faces[i]);

		vector<Rect> mouths = detectMouths(faceROI, faces[i]);
		Rect bestMouth = determineCorrectMouth(mouths, faces[i]);
		mouthPositions = displayMouth(bestMouth, frame, faces[i]);

		return{ nosePositions[0], nosePositions[1], mouthPositions[0], mouthPositions[1] };
	}
	return{0,0,0,0};
}

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

Rect determineCorrectNose(vector<Rect> noses, Rect face) {

	if (noses.size() == 0) {
		//no nose is found
		return Rect(0, 0, 0, 0);
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
		//no mouth below midpoint of face is found
		return Rect(0, 0, 0, 0);
	}

	//get the spatial variance of the mouth locations
	float varianceX = getVariance(newMouths)[0];
	float varianceY = getVariance(newMouths)[1];
	float stdDevX = sqrt(varianceX);
	float stdDevY = sqrt(varianceY);
	float avgX = getAvgXYCenter(newMouths)[0];
	float avgY = getAvgXYCenter(newMouths)[1];

	
	//chop off outliers
	return findSingleBestAttribute(newMouths, avgX, avgY, stdDevX, stdDevY, MOUTH_STANDARD_DEVIATIONS);
}

vector<Rect> detectEyes(Mat faceROI, Rect face) {
	vector<Rect> eyes;
	float width = face.width;
	float height = face.height;
	//-- In each face, detect eyes;
	eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 0, 0 | CASCADE_SCALE_IMAGE, Size(width / 5.0, height / 5.0), Size(width / 3.0, height / 3.0));
	return eyes;
}

vector<Rect> detectNoses(Mat faceROI, Rect face) {
	vector<Rect> noses;
	nose_cascade.detectMultiScale(faceROI, noses, 1.2, 0, 0 | CASCADE_SCALE_IMAGE, Size(50, 50));
	return noses;
}

vector<Rect> detectMouths(Mat faceROI, Rect face) {
	vector<Rect> mouths;
	mouth_cascade.detectMultiScale(faceROI, mouths, 1.2, 0, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	//cout << "NUM MOUTHS: " << mouths.size() << endl;
	return mouths;
}

vector<float> displayNose(Rect nose, Mat frame, Rect face) {

	return{ nose.x / float(face.width) , nose.y / float(face.height) };
}

vector<float> displayMouth(Rect mouth, Mat frame, Rect face) {
	return{ mouth.x / float(face.width) , mouth.y / float(face.height) };
}


//gets the variance of an attribute distribution
//returns vector of size  representing <xVariance, yVariance> 
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
