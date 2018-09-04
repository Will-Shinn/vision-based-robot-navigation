// cannytest.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "camerads.h"
#include "mcl.h"
#include "rand.h"
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv/cxcore.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <opencv/cv.h>
#include <time.h>
#include <opencv/cvaux.h>
#include <cmath>
#include <math.h>
#include <vector>
#include <iostream>
//#include"ros/ros.h"
#include <ctype.h>

using  namespace std;
using namespace cv;


//  multiple video in one single  window
void cvShowMultiImages(char* title, int nArgs, ...)
{

	// img - Used for getting the arguments 
	IplImage* img;

	// DispImage - the image in which all the input images are to be copied
	IplImage* DispImage;

	int size;    // size - the size of the images in the window
	int ind;        // ind - the index of the image shown in the window
	int x, y;    // x,y - the coordinate of top left coner of input images
	int w, h;    // w,h - the width and height of the image

	// r - Maximum number of images in a column 
	// c - Maximum number of images in a row 
	int r;
	int	C;

	// scale - How much we have to resize the image
	float scale;
	// max - Max value of the width and height of the image
	int max;
	// space - the spacing between images
	int space;

	// If the number of arguments is lesser than 0 or greater than 12
	// return without displaying 
	if (nArgs <= 0) {
		printf("Number of arguments too small..../n");
		return;
	}
	else if (nArgs > 12) {
		printf("Number of arguments too large..../n");
		return;
	}
	// Determine the size of the image, 
	// and the number of rows/cols 
	// from number of arguments 
	else if (nArgs == 1) {
		r = C = 1;
		size = 300;
	}
	else if (nArgs == 2) {
		r = 2; C = 1;
		size = 300;
	}
	else if (nArgs == 3 || nArgs == 4) {
		r = 2; C = 2;
		size = 300;
	}
	else if (nArgs == 5 || nArgs == 6) {
		r = 3; C = 2;
		size = 200;
	}
	else if (nArgs == 7 || nArgs == 8) {
		r = 4; C = 2;
		size = 200;
	}
	else {
		r = 4; C = 3;
		size = 150;
	}

	// Create a new 3 channel image to show all the input images
	DispImage = cvCreateImage(cvSize(60 + size*r, 20 + size*c), IPL_DEPTH_8U, 3);

	// Used to get the arguments passed
	va_list args;
	va_start(args, nArgs);

	// Loop for nArgs number of arguments
	space = 20;
	for (ind = 0, x = space, y = space; ind < nArgs; ind++, x += (space + size)) {

		// Get the Pointer to the IplImage
		img = va_arg(args, IplImage*);

		// Check whether it is NULL or not
		// If it is NULL, release the image, and return
		if (img == 0) {
			printf("Invalid arguments");
			cvReleaseImage(&DispImage);
			return;
		}

		// Find the width and height of the image
		w = img->width;
		h = img->height;

		// Find whether height or width is greater in order to resize the image
		max = (w > h) ? w : h;

		// Find the scaling factor to resize the image
		scale = (float)((float)max / size);

		// Used to Align the images
		// i.e. Align the image to next row
		if (ind % r == 0 && x != space) {
			x = space;
			y += space + size;
		}

		// Set the image ROI to display the current image
		cvSetImageROI(DispImage, cvRect(x, y, (int)(w / scale), (int)(h / scale)));

		// Resize the input image and copy the it to the Single Big Image
		cvResize(img, DispImage);

		// Reset the ROI in order to display the next image
		cvResetImageROI(DispImage);
	}

	// Create a new window, and show the Single Big Image
	//cvNamedWindow( title, 1 );
	cvShowImage(title, DispImage);


	// End the number of arguments
	va_end(args);

	// Release the Image Memory
	cvReleaseImage(&DispImage);
}

/*int getpixel(IplImage *image, int x, int y, int *h, int *s, int *v){
	*h = (uchar)image->imageData[y *image->widthStep + x * image->nChannels];
	*s = (uchar)image->imageData[y *image->widthStep + x * image->nChannels + 1];
	*v = (uchar)image->imageData[y *image->widthStep + x * image->nChannels + 2];
	return 0;
}*/

location pitchCopy[PARTICLES];
int      top10Index[LOCS_SORTED] = { 0 };
location top10[LOCS_SORTED];
//map 
float markerCoord[51][2] =
{
	{ 0, 0 },
	{ 0, 100 },
	{ 0, 200 },
	{ 0, 300 },
	{ 0, 400 },
	{ 0, 500 },
	{ 0, 600 },
	{ 100, 0 },
	{ 100, 200 },
	{ 100, 400 },
	{ 100, 600 },
	{ 200, 0 },
	{ 200, 100 },
	{ 200, 200 },
	{ 200, 300 },
	{ 200, 400 },
	{ 200, 500 },
	{ 200, 600 },
	{ 300, 0 },
	{ 300, 100 },
	{ 300, 200 },
	{ 300, 300 },
	{ 300, 400 },
	{ 300, 600 },
	{ 350, 600 },
	{ 400, 0 },
	{ 400, 200 },
	{ 400, 400 },
	{ 400, 600 },
	{ 500, 0 },
	{ 500, 100 },
	{ 500, 200 },
	{ 500, 300 },
	{ 500, 400 },
	{ 500, 500 },
	{ 500, 600 },
	{ 600, 0 },
	{ 600, 200 },
	{ 600, 400 },
	{ 600, 600 },
	{ 700, 0 },
	{ 700, 200 },
	{ 700, 400 },
	{ 700, 600 },
	{ 800, 0 },
	{ 800, 100 },
	{ 800, 200 },
	{ 800, 300 },
	{ 800, 400 },
	{ 800, 500 },
	{ 800, 600 }	
};
motionTypes motionType = JITTER;

// Choose the way new samples are chosen

sampleTypes sampleType = SENSOR_RESETTING;

// Choose the way that we do probabilistic search

searchTypes searchType = MAXPROB_PLUS;

// Values used by sensor resetting plus

double shortAverageProb = 0.0;
double longAverageProb = 0.0;

// We cluster the particles into a set of 2 x 2 x 2 sub-grids of a (conceptual 
// since we don't actually need to build it) 10 x 10 x 10 grid.

clusterBucket subMatrices[10][10][10];
location mostLikelyLocation;
void Mcl::initializePitch()
{


	for (int i = 0; i < PARTICLES; i++)
	{
		pitch[i].xValue = (float)drand48() * MAX_X;
		pitch[i].yValue = (float)drand48() * MAX_Y;
		pitch[i].theta = (float)(M_PI - (drand48() * TWO_PI));
		pitch[i].prob = 1.0 / PARTICLES;
	}

	mostLikelyLocation.xValue = (float)drand48() * MAX_X;
	mostLikelyLocation.yValue = (float)drand48() * MAX_Y;
	mostLikelyLocation.theta = (float)(M_PI - (drand48() * TWO_PI));
	mostLikelyLocation.prob = 1.0 / PARTICLES;

}
//-----------------------------------------------------------------------------
//
// update
//

void Mcl::update(motionData motion, int object, float angle, float distance)
{

  double newProb;

  //OSYSPRINT(("LO> Updating with motion\n"));
  //OSYSPRINT(("LO> Sample set\n"));
  //printPitchUnsort(pitch);

  // Sample a set of particles from the current set of particles

  for (int i=0; i<PARTICLES; i++)
    {
      pitchCopy[i] = pitch[randomSample(pitch,totalProb(pitch))];
    }

  //OSYSPRINT(("LO> Samples selected\n"));
  //printPitchUnsort(pitchCopy);

  // Apply motion update to this set

  for (int i=0; i<PARTICLES; i++)
    {
      pitchCopy[i] = kinematics(pitchCopy[i], motion);
    }

  //OSYSPRINT(("LO> Samples after motion update\n"));
  //printPitchUnsort(pitchCopy);

  // Step through the set of particles, recomputing the probability of
  // each estimated location using the sensor data. Apply the
  // probability filter suggested by the German Team, 2004 report,
  // p44.  (this stops spurious observations messing up a nicely
  // localized robot). Also sum these values as we go along to save
  // time in normalization, and find the max value.

  for (int i=0; i<PARTICLES; i++)
    {
      newProb = computeWeight(pitchCopy[i], object, angle, distance);

      //
      // Switch between these two lines to turn the probability filter
      // on and off.
      //
      //pitchCopy[i].prob = probabilityFilter(guestimate.prob, newProb);

      pitchCopy[i].prob = newProb;

      //
      // End of switch
      //
    }
  
  //OSYSPRINT(("LO> After update with observation\n"));
  //printPitchUnsort(pitchCopy);

  // Here is where we resample, move the particles about in a
  // probabilistic search, replace some samples to cope with kidnapped
  // robots, and normalize...
  //
  // We only normalize at the end, since sample replacement is based on
  // the total probability of the sample set. As a result, we have to pass the
  // total probability as a parameter.

  lowVarianceResample(pitchCopy,totalProb(pitchCopy));
  //OSYSPRINT(("LO> After resampling\n"));
  //printPitchUnsort(pitchCopy);

  probabilisticSearch(pitchCopy, maxProb(pitchCopy));
  //OSYSPRINT(("LO> After probabilistic search\n"));
  //printPitchUnsort(pitchCopy);

  if(PRESERVE_PARTICLES)

    // If we are going to avoid replacing the most likely particles. We need to
    // figure out which ones they are.

    {
      findTop10();
    }

  addRandomElementsTo(pitchCopy, totalProb(pitchCopy));
  //OSYSPRINT(("LO> After adding random elements\n"));
  //printPitchUnsort(pitchCopy);

  normalize(pitchCopy, totalProb(pitchCopy));
  //OSYSPRINT(("LO> After normalization\n"));
  //printPitchUnsort(pitchCopy);
 
  // Finally, copy back locations into pitch.
  
  for(int i=0; i<PARTICLES; i++)
    {
      pitch[i] = pitchCopy[i];  
    }
}
//-----------------------------------------------------------------------------
//
// updateWithMotion
//

void Mcl::updateWithMotion(motionData motion)
{

	loc    guestimate;

	//OSYSPRINT(("LO> Updating with motion\n"));
	//OSYSPRINT(("LO> Sample set\n"));
	//printPitchUnsort(pitch);

	for (int i = 0; i<PARTICLES; i++)
	{

		// Pick a random particle from the current location
		// representation as an estimated location and update this with
		// the latest motion information.

		guestimate = pitch[randomSample(pitch, totalProb(pitch))];

		//OSYSPRINT(("LO> Selection is\n"));
		//OSYSPRINT(("LO> xValue %5d\n", (int)guestimate.xValue));
		//OSYSPRINT(("LO> yValue %5d\n", (int)guestimate.yValue));
		//OSYSPRINT(("LO> theta  %5f\n", guestimate.theta));

		guestimate = kinematics(guestimate, motion);

		// Save revised estimate into new set of particles:

		pitchCopy[i] = guestimate;
	}

	// Finally, write the new location representation into pitch, keeping
	// probability the same. 

	for (int i = 0; i<PARTICLES; i++)
	{
		pitch[i] = pitchCopy[i];
	}

	//OSYSPRINT(("LO> Samples after motion update\n"));
	//printPitchUnsort(pitch);

}
//-----------------------------------------------------------------------------
//
// randomSample   随机取样
//
// Function which picks a random location from array of locations
// whose probability sum up to totalProb, and returns the index of
// that location.
//

int Mcl::randomSample(loc* array, double totalProb)
{
	int    location;
	double prob;

	// Generate a probability using the approved random number generator.
	// scaling appropriately.

	prob = (double)drand48() * totalProb;

	//OSYSPRINT(("LO> Sample location with probability %lf\n", prob));

	// Find the location identified by this probability

	location = pickSample(prob, array);

	return location;
}


int Mcl::pickSample(double prob, loc* array)
{
	double cumulative = 0.0;
	int   i = 0;

	while ((i < PARTICLES) && (cumulative < prob))
	{
		cumulative += array[i].prob;
		i++;
	}

	// If we have found a location, return it

	if (cumulative >= prob)
	{
		return i - 1; // Betsy's brilliant solution, a result of years of
		// experience.
	}

	// Otherwise there must be a problem, so pick a random location
	// Don't rely on this working well, okay, it is just meant to try
	// to avoid dumb crashes.

	else
	{

		printf(("LO> pickSample failing to find sensible sample\n"));

		return (int)drand48() * PARTICLES;
	}
}
//-----------------------------------------------------------------------------
//
// totalProb
// Function which sums the probabilities of a set of particles. 
//
double Mcl::totalProb(loc* aPitch)
{
	double cumulative = 0;
	int   i = 0;

	for (int i = 0; i<PARTICLES; i++)
	{
		cumulative += aPitch[i].prob;
	}

	return cumulative;
}

//-----------------------------------------------------------------------------
//
// kinematics 动作模块
//Given data on what motion the robot is trying to
// execute and where it was last time we knew (or thought we knew), guess
// where we are now.
// connect parts to ROS 
loc Mcl::kinematics(loc guestimate, motionData motion)
{
	locData medianMovement;
	locData actualMovement;
	locData globalMovement;
	loc     newPosition;

	// Input motion is an estimate of the movement of the robot in the last
	// time period.

	//OSYSPRINT(("LO> Recent motion\n"));
	//OSYSPRINT(("LO> xValue %f\n", motion.xValue));
	//OSYSPRINT(("LO> yValue %f\n", motion.yValue));
	//OSYSPRINT(("LO> theta  %f\n", motion.theta));

	// Motion is subject to error, and we need to take that into account.
	// We do this by Monte-Carlo; picking a sample movement and using that.

	actualMovement = pickRandomMotion(motion);

	// This movement is relative to the robot; we need to convert to the 
	// global frame and combine this with guestimate in order to have a
	// new value to return.

	// Need to check that the angles are all measured in the same direction.

	globalMovement = changeFrameLocalToGlobal(actualMovement, guestimate);
	newPosition.xValue = guestimate.xValue + globalMovement.xValue;
	newPosition.yValue = guestimate.yValue + globalMovement.yValue;
	newPosition.theta =
		adjustForCorrectMean(compensateForFullTurn(guestimate.theta
		+ globalMovement.theta));
	//OSYSPRINT(("LO> newPosition\n"));
	//OSYSPRINT(("LO> xValue %5d\n", (int)newPosition.xValue));
	//OSYSPRINT(("LO> yValue %5d\n", (int)newPosition.yValue));
	//OSYSPRINT(("LO> theta  %5f\n", newPosition.theta));

	// Not sure that this value is right... probably need to factor in how
	// likely the movement is, but even without that, this process will
	// spread out the particles just as it should.

	newPosition.prob = guestimate.prob;

	return newPosition;
}
//-----------------------------------------------------------------------------
//
// pickRandomMotion 给之前的输入信号加上高斯噪音
// random motion sample assuming x  y and theta data
//plus the nosie to the input part above
locData Mcl::pickRandomMotion(motionData mean)
{
	locData sample, stdev, increment, offset, movement;

	// We set the standard deviation of the error distribution to be
	// proportional to the distance we think we have moved, but if that
	// is zero, may add a small amount of jitter. Because of the way it
	// is used, we make stdev positive.

	if (motionType == JITTER)
	{
		if (mean.xValue != 0)
		{
			stdev.xValue = fabs(X_MOTION_STDEV * mean.xValue);
		}
		else
		{
			stdev.xValue = X_ZERO_MOTION_STDEV;
		}

		if (mean.yValue != 0)
		{
			stdev.yValue = fabs(Y_MOTION_STDEV * mean.yValue);
		}
		else
		{
			stdev.yValue = Y_ZERO_MOTION_STDEV;
		}

		if (mean.theta != 0)
		{
			stdev.theta = fabs(A_MOTION_STDEV * mean.theta);
		}
		else
		{
			stdev.theta = A_ZERO_MOTION_STDEV;
		}
	}
	else
	{
		stdev.xValue = fabs(X_MOTION_STDEV * mean.xValue);
		stdev.yValue = fabs(Y_MOTION_STDEV * mean.yValue);
		stdev.theta = fabs(A_MOTION_STDEV * mean.theta);
	}

	//OSYSPRINT(("LO> Stdev\n"));
	//OSYSPRINT(("LO> xValue %5f\n", stdev.xValue));
	//OSYSPRINT(("LO> yValue %5f\n", stdev.yValue));
	//OSYSPRINT(("LO> theta  %5f\n", stdev.theta));

	// Sample from the motion error distribution: for each of x, y and
	// theta, using the formula on p124 of Thrun et al.

	offset.xValue = sampleFromGaussian(stdev.xValue);
	offset.yValue = sampleFromGaussian(stdev.yValue);
	offset.theta = sampleFromGaussian(stdev.theta);

	//OSYSPRINT(("LO> Offset\n"));
	//OSYSPRINT(("LO> xValue %5f\n", offset.xValue));
	//OSYSPRINT(("LO> yValue %5f\n", offset.yValue));
	//OSYSPRINT(("LO> theta  %5f\n", offset.theta));

	// Make the error adjustment

	movement.xValue = mean.xValue + offset.xValue;
	movement.yValue = mean.yValue + offset.yValue;
	movement.theta =
		adjustForCorrectMean(compensateForFullTurn(mean.theta + offset.theta));

	//OSYSPRINT(("LO> Adjustment\n"));
	//OSYSPRINT(("LO> xValue %5f\n", movement.xValue));
	//OSYSPRINT(("LO> yValue %5f\n", movement.yValue));
	//OSYSPRINT(("LO> theta  %5f\n", movement.theta));

	return movement;
}
//-----------------------------------------------------------------------------
//
// sampleFromGaussian给之前的噪音算法提供高斯噪音
//
// Given the stdev of a Gaussian, pick a sample from it.

float Mcl::sampleFromGaussian(float stdev)
{

	double  sample = 0;

	for (int i = 0; i<12; i++)
	{
		sample += ((2 * stdev * drand48()) - stdev);
	}

	sample *= 0.5;

	return (float)sample;
}
//-----------------------------------------------------------------------------
//
// changeFrameLocalToGlobal 转换局部变量到全局变量
//
// Given a movement in the local frame, and the angle of orientation of the
// robot, compute the movement in the global frame.
//
// This is R(theta)^{-1} from page 53 of Siegwart and Nourbakhsh.
//
// Of course it isn't precise, since the formula is only correct for 
// instantaneous motion, but, hey, we can't be perfect.

locData Mcl::changeFrameLocalToGlobal(locData localMovement, loc position)
{
	locData globalMovement;

	globalMovement.xValue = localMovement.xValue * cos(position.theta)
		- localMovement.yValue * sin(position.theta);
	globalMovement.yValue = localMovement.xValue * sin(position.theta)
		+ localMovement.yValue * cos(position.theta);
	globalMovement.theta = localMovement.theta;

	//OSYSPRINT(("LO> globalMovement\n"));
	//OSYSPRINT(("LO> xValue %5f\n", globalMovement.xValue));
	//OSYSPRINT(("LO> yValue %5f\n", globalMovement.yValue));
	//OSYSPRINT(("LO> theta  %5f\n", globalMovement.theta));

	return globalMovement;
}
//-----------------------------------------------------------------------------
//
// checkXBounds
//因为范围有限，通过限制边界来剔除不可能的位置
// Since (a) we are sampling from an infinite distance distribution and (b)
// may think we are moving beyond the pitch boundaries, give localization a
// little help by ruling out impossible positions.

float Mcl::checkXBounds(float xValue)
{

	if (xValue > (MAX_X + 1000))
	{
		xValue = MAX_X;
	}
	else
	if (xValue < MIN_X)
	{
		xValue = MIN_X;
	}

	return xValue;
}

//-----------------------------------------------------------------------------
//
// checkYBounds
//

float Mcl::checkYBounds(float yValue)
{

	if (yValue > MAX_Y)
	{
		yValue = MAX_Y;
	}
	else
	if (yValue < MIN_Y)
	{
		yValue = MIN_Y;
	}

	return yValue;
}
//-----------------------------------------------------------------------------
//
// compensateForFullTurn
//角度补偿 ，通过对角度可能情况的补偿，来限制其在0到2PI之间
// If we keep incrementing the angle component, we end up with large numbers
// that are hard to interpret. So restrict the range to 0 - 2PI.
//
// For values larger than 2PI this is easy; we assume it is less than
// 4PI and so just take off 2PI. If the value is less than 0, we can
// assume it will be between 0 and -2PI and so just add 2PI.

float Mcl::compensateForFullTurn(float theta)
{

	if (theta > TWO_PI)
	{
		theta = theta - TWO_PI;
	}
	else
	if (theta < 0)
	{
		theta = theta + TWO_PI;
	}

	return theta;
}
//----------------------------------------------------------------------------
//
// updateWithObservation
//
// For every particle, update the probability that the robot is in that 
// location given the sensor reading.
void Mcl::updateWithObservation(int object, float angle, float distance)
{
	loc    guestimate;
	double newProb;

	printf(("LO> =========== Update with Observation ===========\n"));

	// Step through the set of particles, recomputing the probability of
	// each estimated location using the sensor data. Apply the
	// probability filter suggested by the German Team, 2004 report,
	// p44.  (this stops spurious observations messing up a nicely
	// localized robot). Also sum these values as we go along to save
	// time in normalization, and find the max value.

	for (int i = 0; i<PARTICLES; i++)
	{
		guestimate = pitch[i];
		newProb = computeWeight(guestimate, object, angle, distance);

		//
		// Switch between these two lines to turn the probability filter
		// on and off.
		//
		//guestimate.prob = probabilityFilter(guestimate.prob, newProb);

		guestimate.prob = newProb;

		//
		// End of switch
		//

		// Save revised estimate into new set of particles:

		pitchCopy[i] = guestimate;
	}

	printf(("LO> After update with observation\n"));
	printPitchUnsort(pitchCopy);

	// Here is where we resample, move the particles about in a
	// probabilistic search, replace some samples to cope with kidnapped
	// robots, and normalize...
	//
	// We only normalize at the end, since sample replacement is based on
	// the total probability of the sample set. As a result, we have to pass the
	// total probability as a parameter.

	lowVarianceResample(pitchCopy, totalProb(pitchCopy));
	printf(("LO> After resampling\n"));
	printPitchUnsort(pitchCopy);

	//probabilisticSearch(pitchCopy, maxProb(aPitch));
	//OSYSPRINT(("LO> After probabilistic search\n"));
	//printPitchUnsort(pitchCopy);

	if (PRESERVE_PARTICLES)

		// If we are going to avoid replacing the most likely particles. We need to
		// figure out which ones they are.

	{
		findTop10();
	}

	addRandomElementsTo(pitchCopy, totalProb(pitchCopy));
	printf(("LO> After adding random elements\n"));
	printPitchUnsort(pitchCopy);

	normalize(pitchCopy, totalProb(pitchCopy));
	printf(("LO> After normalization\n"));
	printPitchUnsort(pitchCopy);

	// Finally, copy back locations into pitch.

	for (int i = 0; i<PARTICLES; i++)
	{
		pitch[i] = pitchCopy[i];
	}

}
//-----------------------------------------------------------------------------
//
// computeWeight 计算权重
//
// Sensor model; tells us how likely we are to be at a particular location,
// given what we see and where we see it. 
//

double Mcl::computeWeight(loc guestimate, int object, float angle, float dist)
{

	polData measure, mean, stdev;

	// The correct weight for guestimate is provided by the 2D gaussian. 

	// First we compute the distance and angle to the object that we see
	// given the position in guestimate (which is straightforward given
	// that we know the absolute location of every object) by applying
	// the global to local coordinate transformation. This gives us the
	// median values for the 2D gauassian

	mean = computeDistance(guestimate, object);

	// Stdev is computed as suggested by Fox; constant for angle and 
	// proportional to measurement for distance.

	stdev.radius = R_SENSOR_STDEV * dist;
	stdev.theta = T_SENSOR_STDEV;

	// Now we use the 2D gaussian to give us the probability of seeing the 
	// object that we saw given our guestimated position.

	measure.radius = dist;
	measure.theta = angle;

	// Compute weight as according to Thrun et al.

	guestimate.prob = gaussian2D(measure, mean, stdev);

	//OSYSPRINT(("LO> Probability of location %5d %5d %5f is %7lf\n", (int)guestimate.xValue,(int)guestimate.yValue,(float)guestimate.theta,(float)guestimate.prob));
	return guestimate.prob;
}
//-----------------------------------------------------------------------------
//
// computeDistance 计算距离
//

// Compute the distance and bearing of an object (whose position I know) from 
// myLocation.

polData Mcl::computeDistance(loc myLocation, int object)
{
  
  polData relativePosition; // Relative location of marker to myLocation

  float objectX;            // Global coordinates of the markers
  float objectY;

  // Look-up the global x, y positions of the marker in the table of marker 
  // locations.

  objectX = markerCoord[object][0];
  objectY = markerCoord[object][1];

  // Distance to marker from dog comes from simple trigonometry...

  relativePosition.radius = (float)sqrt(pow((objectX - myLocation.xValue),2) + 
                                        pow((objectY - myLocation.yValue),2));

  // Angle also comes from simple trigonometry, but is a bit more complex to
  // get right. 通过三角函数
  //
  // First we compute the angle of the marker relative to the dog
  // assuming that the dog is facing north. To do this, we need to
  // take care to get the angle in the right quadrant. Remember that
  // north is 0/2PI radians, and we measure the angle clockwise
  // from N (so that if the robot is in the middle of the pitch and
  // facing north, PB is PI/4, PY is 3PI/4, YP is 5PI/4 and BP is
  // 7PI/4.

  if ((objectX >= myLocation.xValue) && (objectY >= myLocation.yValue))
    {
      // Theta is between 0 and PI/2.

      relativePosition.theta = (float)atan2((objectY - myLocation.yValue),
                                            (objectX - myLocation.xValue));

    }
  else 
  if ((objectX < myLocation.xValue) && (objectY >= myLocation.yValue))
    {
      // Theta is between PI/2 and PI.

      relativePosition.theta = (float) atan2((myLocation.xValue - objectX),
                                             (objectY - myLocation.yValue));
      relativePosition.theta  = relativePosition.theta  + HALF_PI;
    }
  else
  if ((objectX < myLocation.xValue) && (objectY < myLocation.yValue))
    {
      // Theta is between PI/2 and 3PI/2.

      relativePosition.theta = (float)atan2((myLocation.yValue - objectY),
                                            (myLocation.xValue - objectX));
      relativePosition.theta  = relativePosition.theta  + M_PI;
    }
  else 
    // ((objectX >= myLocation.xValue) && (objectY < myLocation.yValue))
    {
      // Theta is between 3PI/2 and 2PI.

      relativePosition.theta = (float)atan2((myLocation.yValue - objectY),
                                            (objectX - myLocation.xValue));
      relativePosition.theta  *= -1;
      relativePosition.theta  = relativePosition.theta  + TWO_PI;
    }

  // This angle is the angle to the marker if the dog is facing north
  // at myLocation, so we have to adjust to get the bearing between
  // the dog and the marker. Turns out we can do that just like this:

  relativePosition.theta 
    = adjustForCorrectMean
	 (compensateForFullTurn(relativePosition.theta - myLocation.theta));

  //OSYSPRINT(("LO> My location is %d %d %f\n", (int)myLocation.xValue,(int)myLocation.yValue,myLocation.theta));
  //OSYSPRINT(("LO> Marker %d is at %f %f\n", object, objectX, objectY));
  //OSYSPRINT(("LO> Relative location is %f %f\n", relativePosition.radius, relativePosition.theta));
  
  return relativePosition;
  
}

//-----------------------------------------------------------------------------
//
// adjustForCorrectMean
//
// We measure the heading of the dog between 0 and TWO_PI, but since
// we compute weights using a zero-centred gaussian, we need to turn
// values between PI and TWO_PI into values between -PI and 0.

float Mcl::adjustForCorrectMean(float theta)
{

	if (theta > M_PI && theta <=TWO_PI)
	{
		theta = theta - TWO_PI;
	}

	return theta;
}
//-----------------------------------------------------------------------------
//
// gaussian2D 2D 高斯函数
//用作更新传感器并且计算计算可能性
// 2Dguassian is used for sensor update, and computes the probability
// that the robot is at measure.radius and measure.theta from the
// marker (in polar co-ordinates) given that the world model says it
// it should be mean.radius and mean.theta from the marker.

double Mcl::gaussian2D(polData measure, polData mean, polData stdev)
{

	double p = 0;
	double q = 0;

	// Prevent divide by zero.

	if (stdev.radius == 0)
	{
		stdev.radius = 0.001;
	}

	if (stdev.theta == 0)
	{
		stdev.theta = 0.001;
	}

	// Now do 2D gaussian calculation. This follows line 5 of Table 6.4
	// on p179 of Thrun et al. The 2D result is the product of two 1D
	// computations, one for distance and one for angle.

	p = (1 / (sqrt(2 * M_PI*stdev.radius*stdev.radius))) *
		exp(-0.5*(pow((measure.radius - mean.radius), 2)) / (2 * pow(stdev.radius, 2)));

	q = (1 / (sqrt(2 * M_PI*stdev.theta*stdev.theta))) *
		exp(-0.5 * (pow((measure.theta - mean.theta), 2)) / (2 * pow(stdev.theta, 2)));

	//OSYSPRINT(("LO> p is %7lf and q is %7lf\n", p, q));

	return p * q;
}
//-----------------------------------------------------------------------------
//
// probabilityFilter
//
// The German team probability filter---don't allow the probability to change
// too much.限制可能性过大的变化，排除一些奇点


double Mcl::probabilityFilter(double oldProb, double newProb)
{
	if (newProb > oldProb)
	{

		// Maximum increment allowed

		if (newProb - oldProb > 0.1)
		{
			return oldProb + 0.1;
		}
		else
		{
			return newProb;
		}
	}
	else
	{

		// Maximum decrement allowed.

		if (oldProb - newProb > 0.05)
		{
			return oldProb - 0.05;
		}
		else
		{
			return newProb;
		}

	}
}
//-----------------------------------------------------------------------------
//
// lowVarianceResample 
//
// A more sophisticated resampler, taken from Thrun et
// al. p110. Variable names reflect those used in the Thrun et
// al. algorithm.
//
// Everything here is scaled by the total probability of the sample
// set since we call this on the unormalizaed values.

void Mcl::lowVarianceResample(loc* aPitch, double totalProb)
{
	loc    otherPitchCopy[PARTICLES];

	double    rProb = (drand48() / PARTICLES) * totalProb;
	double    uProb = rProb;
	double    increment = 1/PARTICLES * totalProb;
	int       selection;

	// Build a set of samples, by picking one that corresponds to myRand
	// then one that corresponds to myRand + 1/PARTICLES and so on.

	for (int i = 0; i<PARTICLES; i++)
	{
		otherPitchCopy[i] = aPitch[pickSample(uProb, aPitch)];
		uProb = rProb + (i * increment);
	}

	// ... then copy those elements back into aPitch, the parameter that
	// the function was called with.

	for (int i = 0; i<PARTICLES; i++)
	{
		aPitch[i] = otherPitchCopy[i];
	}
}
//-----------------------------------------------------------------------------
//
// probabilisticSearch
//
// Do a probabilistic search from the current set of particles.
//
// This is similar to adding random elements, but we base each search step on
// existing particles, moving randomly from them, a small bit for particles
// that are more probable, and moving more from particles that are less
// probable.
//
// The way we do this is drawn from the method used by the German team (2004 
// report, page 46).
//
// I suspect that directly using the GT approach will cause a lot of
// jitter; our particles have, I suspect, much lower probabilities
// than theirs because of the way that these are calculated, so we try
// basing the offset on divergence from the probability of the most likely
// particle.

void Mcl::probabilisticSearch(loc* aPitch, double maxProb)
{
	for (int i = 0; i<PARTICLES; i++)
	{

		switch (sampleType)
		{

		case GERMAN:

			// The straight GT approach.
			//
			// The (-1 + (2 * drand48)) just generates a number
			// between -1 and 1 since there is no library function to
			// do this.

			aPitch[i].xValue += (float)100 * (1 - aPitch[i].prob)
				* (-1 + (2 * drand48()));
			aPitch[i].yValue += (float)100 * (1 - aPitch[i].prob)
				* (-1 + (2 * drand48()));
			aPitch[i].theta += (float) .5 * (1 - aPitch[i].prob)
				* (-1 + (2 * drand48()));
			break;

		case MAXPROB:

			// The maxProb approach
			// 
			// Since our max probabilities are much lower than the GT
			// seem to generate, use the maximum probability we find
			// in the particle distribution as an upper limit and multiply
			// by a bigger factor.

			aPitch[i].xValue += (float)1000 * (-1 + (2 * drand48()))
				* (maxProb - aPitch[i].prob);
			aPitch[i].yValue += (float)1000 * (-1 + (2 * drand48()))
				* (maxProb - aPitch[i].prob);
			aPitch[i].theta += (float)5 * (-1 + (2 * drand48()))
				* (maxProb - aPitch[i].prob);
			break;

		case MAXPROB_PLUS:

			// The maxProb approach with a bound on the largest change
			// you can make which keeps to the limits used by the GT.

			aPitch[i].xValue += (float)1000 * (-1 + (2 * drand48()))
				* max(0.1, (maxProb - aPitch[i].prob));
			aPitch[i].yValue += (float)1000 * (-1 + (2 * drand48()))
				* max(0.1, (maxProb - aPitch[i].prob));
			aPitch[i].theta += (float)5 * (-1 + (2 * drand48()))
				* max(0.1, (maxProb - aPitch[i].prob));
			break;

		}

		// Whichever method we use, we need to make sure that the angle 
		// doesn't exceed 2PI

		//aPitch[i].prob =
		//	adjustForCorrectMean(compensateForFullTurn(aPitch[i].prob));

	}
}
//-----------------------------------------------------------------------------
//
// addRandomElementsTo
//
// Add some random elements to aPitch.
//
// Note that this is probably not the best way to do things; we stand
// a chance of writing over good samples, and if we are adapting the
// number of particles based on how sure we are of our position
// (adaptive sampling) we will always want to add our random samples
// onto the end of the list.

void Mcl::addRandomElementsTo(loc* aPitch, double totalProb)
{
	int   numberOfRandomSamples, randomIndex;
	int   skippedSamples = 0;
	float averageProb;

	// Figure out how many elements of aPitch we will replace and
	// prepare to set their probability to the average of the distribution.

	numberOfRandomSamples = howManySamples(aPitch, totalProb);
	averageProb = (float)totalProb / PARTICLES;

	printf("LO> Adding %d random samples\n", numberOfRandomSamples);

	// For each sample in numberOfRandomSamples, pick a random
	// element of aPitch, and generate random values for it. Since I can't figure
	// what a good probability would be, just make this the current average
	// probability.

	for (int i = 0; i<numberOfRandomSamples; i++)
	{
		randomIndex = (int)floor(PARTICLES * drand48());

		// notInTop10 preserves the most likely LOCS_PRESERVED particles
		// when the PRESERVE_PARTICLES switch is set.

		if (notInTop10(randomIndex))
		{
			aPitch[randomIndex].xValue = MAX_X * drand48();
			aPitch[randomIndex].yValue = MAX_Y * drand48();
			aPitch[randomIndex].theta = TWO_PI * drand48();
			aPitch[randomIndex].prob = averageProb;
		}
		else
		{
			skippedSamples++;
		}
	}

	//OSYSPRINT(("LO> Skipped samples %d\n",skippedSamples));

}

//-----------------------------------------------------------------------------
//
// howManySamples 
//
// How many new random samples to add in to the current set of particles. 
// Not the most efficient way to do things, but elegant and flexible :-)

int Mcl::howManySamples(loc* aPitch, double totalProb)
{
	double averageProb;

	switch (sampleType)
	{

	case SENSOR_RESETTING:

		// Sensor resetting, as described by Gutmann and Fox

		averageProb = totalProb / PARTICLES;
		return (int)floor(PARTICLES * max(0, (1 - (averageProb / P_THRESHOLD))));
		break;

	case SENSOR_RESETTING_PLUS:

		// Augmented sensor resetting, as described by Gutmann and Fox

		averageProb = totalProb / PARTICLES;
		longAverageProb += ETA_LONG * (averageProb - longAverageProb);
		shortAverageProb += ETA_SHORT * (averageProb - shortAverageProb);
		return (int)floor(PARTICLES
			* max(0, (1 - NU * (shortAverageProb / longAverageProb))));
		break;

	default:

		// Just replace a small constant fraction of the samples

		return (int)floor(PARTICLES*RAND_PROPORTION);

	}
}
//-----------------------------------------------------------------------------
//
// max
//
// We need this since I can't find a library function that does it (probably
// just not looking hard enough).

double Mcl::max(double one, double two)
{
	if (one >= two)
	{
		return one;
	}
	else
	{
		return two;
	}
}

//-----------------------------------------------------------------------------
//
// normalize
//
// Normalize probabilities in an array of locations. 
//
// Note that this is not the correct way to normalize---what you
// should do is to establish the sum of the distribution by sampling
// so that you weight more probable samples appropriately higher, and
// you shouldn't double count samples for the same location.
//
// I guess that this is a reasonable approximation however. 


loc* Mcl::normalize(loc* pitch, double normalizationConstant)
{

	//OSYSPRINT(("LO> -------- Normalizing ------------\n"));

	if (normalizationConstant != 0)
	{
		for (int i = 0; i<PARTICLES; i++)
		{

			//OSYSPRINT(("LO> Location %5f %5f\n", pitch[i].xValue, pitch[i].yValue));
			//OSYSPRINT(("LO> Probability before normalizing %5lf\n", pitch[i].prob));

			pitch[i].prob = pitch[i].prob / normalizationConstant;

			//OSYSPRINT(("LO> Probability after normalizing %5f\n", pitch[i].prob));

		}
	}

	return pitch;
}

//-----------------------------------------------------------------------------
//
// whereAmI
//
// The function that is called from Localization.cc to determine what the
// most likely robot location is.
//
// Just looks through pitch until it find the location with the highest
// probability.

loc Mcl::whereAmI()
{

	double prob;
	int    likelyLocationIndex;
	loc    likelyLocation;

	prob = 0;

	for (int i = 0; i<PARTICLES; i++)
	{
		if (pitch[i].prob >= prob)
		{
			prob = pitch[i].prob;
			likelyLocationIndex = i;
		}
	}

	mostLikelyLocation = pitch[likelyLocationIndex];

	return mostLikelyLocation;
}
//-----------------------------------------------------------------------------
//
// findTop10
//
// Finds the LOCS_SORTED most likely robot locations. Goes through
// all the cells in pitch and picks the most probable cells, and then
// returns them.
//
// This was required for the basic particle localization (at least the
// way that Simon wrote it) and is now mainly on hand to allow us to
// avoid replacing these particles if required.

loc* Mcl::findTop10()
{
	double prob = 0;
	int    x;

	// Initialize top10Index

	for (int i = 0; i<LOCS_SORTED; i++)
	{
		top10Index[i] = 0;
	}

	// Fill an array of top10 most likely locations

	for (int i = 0; i<PARTICLES; i++)
	{
		// Try to insert the ith prob in the top10 array.
		//
		// Start from bottom (smallest) in top10 array and find location
		// to insert this prob value

		x = (LOCS_SORTED - 1);
		while ((pitchCopy[i].prob > pitchCopy[top10Index[x]].prob) && (x >= 0))
		{
			x--;
		}
		x++;

		// if x==LOCS_SORTED, then this prob value is smaller
		// than smallest in top10, so ignore it.
		//
		// otherwise, insert it into top10 in selected spot

		if (x<LOCS_SORTED)
		{
			for (int xx = (LOCS_SORTED - 1); xx>x; xx--)
			{
				top10Index[xx] = top10Index[xx - 1];
			}

			top10Index[x] = i;

		}
	}

	// Now we have the indices of the top 10 locations, we need to return
	// an array that includes these locations.

	for (int i = 0; i<LOCS_SORTED; i++)
	{
		top10[i] = {0};
		top10[i] = pitchCopy[top10Index[i]];
	}

	return top10;
}

//-----------------------------------------------------------------------------
//
// getTime
//

double Mcl::getTime(void)
{
	


		//获取系统的UTC时间。   


		SYSTEMTIME time;
	GetSystemTime(&time);
	return time.wSecond * 1000 + time.wSecond / 1000;
}

//-----------------------------------------------------------------------------
//
// returnMostLikelyLocation
//
// Accessor function for where the robot thinks it is most likely to
// be. Note that it doesn't compute anything, just returns the
// location that was last computed.

loc Mcl::returnMostLikelyLocation(void)
{
	return mostLikelyLocation;
}

//-----------------------------------------------------------------------------
//
// printMostLikelyLocation
//
// Accessor function to print where the robot thinks it is most likely
// to be. Note that it doesn't compute anything, just returns the
// location that was last computed.

loc Mcl::printMostLikelyLocation(void)
{
	printf((" Most likely location <%d %d %f %lf>\n",
		(int)mostLikelyLocation.xValue,
		(int)mostLikelyLocation.yValue,
		(float)mostLikelyLocation.theta,
		(double)mostLikelyLocation.prob));

}

//============================================================================
//
// particle clustering 粒子分类
//
// This is the main thing to add to move to German/UTA-style localization
//

//-----------------------------------------------------------------------------
//
// whereAmIReally
//
// The function that is called from Localization.cc to determine what the
// most likely robot location is. A replacement for whereAmI.
//
// We do the computation of the most likely position here rather than when
// we update the pitch distribution so that we can call it when we want it 
// (from Behavior for example) rather than only updating when we have an 
// observation.
//
// The location that has been computed as most likely,mostLikelyLocation, is
// global within mcl, so we can ask for it without having to recompute.

loc Mcl::whereAmIReally()
{
	//OSYSPRINT(("LO> Finding most likely location\n"));
	findMostLikelyLocation();
	return mostLikelyLocation;
}
//-----------------------------------------------------------------------------
//
// findMostLikelyLocation
//
// 
//
// Clusters particles to establish a more reliable estimate of position than
// whereAmI did; one that does not bounce around since it averages over more
// possible locations.


void Mcl::findMostLikelyLocation()
{

	loc likelyLocation;

	//printAPitch();

	initialiseMatrices();
	clusterParticles(pitch); // This call builds the set of subMatrices
	// and does most of the work.

	// This function establishes the value of mostLikelyLocation

	findBestSubMatrix();

}
// ---------------------------------------------------------------------------- -
//
// clusterParticles
// 
// Go through the set of particles, placing them into buckets; there are 10
// buckets in each of the x, y, theta directions.
//
// At the same time we do this, we place them into the relevant bucket
// in the set of 2 x 2 x 2 submatrices which we search to find the
// best cluster. We do this as a combined step to avoid having to pop
// each element out of the bucket in clusterMatrix in order to put it in
// the bucket in the submatrix list.
//
// As a result, we don't need clusterMatrix.

void Mcl::clusterParticles(loc* aPitch)
{

	int           xIncrement;
	int           yIncrement;
	float         thetaIncrement;
	locIndex      bucketIndex;

	xIncrement = MAX_X / 10;
	yIncrement = MAX_Y / 10;
	thetaIncrement = (2 * M_PI) / 10;

	//OSYSPRINT(("LO> Clustering particles\n"));

	for (int i = 0; i<PARTICLES; i++)
	{

		for (int j = 1; j <= 10; j++)
		for (int k = 1; k <= 10; k++)
		for (int L = 1; L <= 10; L++)
		{

			// Stop when we find the upper boundary of the relevant
			// cluster.

			if ((aPitch[i].xValue >  ((j - 1) * xIncrement)) &&
				(aPitch[i].xValue <= (j * xIncrement)) &&
				(aPitch[i].yValue >  ((k - 1) * xIncrement)) &&
				(aPitch[i].yValue <= (k * xIncrement)) &&
				(aPitch[i].theta  >  ((L - 1) * thetaIncrement)) &&
				(aPitch[i].theta <= (L * thetaIncrement)))
			{

				// Add the particle to all the relevant sub-matrices,
				// sending an index between 1 and 10.

				//OSYSPRINT(("LO> bucket indices are %d, %d %d\n", j, k, L));

				bucketIndex.xValue = j;
				bucketIndex.yValue = k;
				bucketIndex.theta = L;

				addToSubMatrix(aPitch[i], bucketIndex);

			}
		}
	}
}


//-----------------------------------------------------------------------------
//
// initialiseMatrices
//
// Create blank versions of both the 10x10x10 clusterMatrix and the 9 x 9 x 9
// matrix that holds the reclustering into 2 x 2 x 2 buckets.
//

void Mcl::initialiseMatrices()
{

	//OSYSPRINT(("LO> Initialising matrices...\n"));

	for (int j = 0; j <= 9; j++)
	for (int k = 0; k <= 9; k++)
	for (int l = 0; l <= 9; l++)
	{
		subMatrices[j][k][l].xValue = 0;
		subMatrices[j][k][l].yValue = 0;
		subMatrices[j][k][l].theta1 = 0;
		subMatrices[j][k][l].theta2 = 0;
		subMatrices[j][k][l].count = 0;
		subMatrices[j][k][l].prob = 0;
	}
}
//-----------------------------------------------------------------------------
//
// addToSubMatrix
//赋至小矩阵中
// Given a particle and the indices that gives its position in the 10
// x 10 x 10 grid, it is easy, but tedious, to add it to the set of 2
// x 2 x 2 sub-grids.
//
// The incoming index will be 1 to 10, relating to the conceptual
// underlying 10 x 10 x 10 grid.
//
// Values are summed like this so that averaging can follow the
// approach given on p46 of the 2004 German Team report

void Mcl::addToSubMatrix(loc particle, locIndex ind)
{

	int j, k, l;

	//OSYSPRINT(("LO> Adding particles to submatrices\n"));
	//OSYSPRINT(("LO> Element %d, %d, %d\n", ind.xValue, ind.yValue, ind.theta));


	// For testing, let's pretend all the elements end up in one subMatrix
	// bucket.

	/*
	subMatrices[1][1][1].xValue += particle.xValue;
	subMatrices[1][1][1].yValue += particle.yValue;
	subMatrices[1][1][1].theta1 += sin(particle.theta);
	subMatrices[1][1][1].theta2 += cos(particle.theta);
	subMatrices[1][1][1].prob   += particle.prob;
	subMatrices[1][1][1].count++;

	}
	*/


	// There are 27 cases.

	//
	// Case 1
	//
	// Not in a boundary cell of the 10 x 10 x 10 for any index

	if (ind.xValue > 1 && ind.xValue <= 9
		&& ind.yValue >  1 && ind.yValue <= 9
		&& ind.theta  > 1 && ind.theta <= 9)
	{
		// Add the particle to the submatrices with index 1 and 2 below
		// each of the indices.

		for (int j = (ind.xValue - 2); j <= (ind.xValue - 1); j++)
		for (int k = (ind.yValue - 2); k <= (ind.yValue - 1); k++)
		for (int l = (ind.theta - 2); l <= (ind.theta - 1); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 2
	//
	// In a lower boundary cell of the 10 x 10 x 10 for x index only

	if (ind.xValue == 1
		&& ind.yValue >  1 && ind.yValue <= 9
		&& ind.theta  > 1 && ind.theta <= 9)
	{
		// Add the particle to the submatrices with index 1 below the x
		// index and 1 and 2 below the other indices.

		for (int j = (ind.xValue - 1); j <= (ind.xValue - 1); j++)
		for (int k = (ind.yValue - 2); k <= (ind.yValue - 1); k++)
		for (int l = (ind.theta - 2); l <= (ind.theta - 1); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 3
	//
	// In an upper boundary cell of the 10 x 10 x 10 for x index only

	if (ind.xValue == 10
		&& ind.yValue >  1 && ind.yValue <= 9
		&& ind.theta  > 1 && ind.theta <= 9)
	{
		// Add the particle to the submatrices with index 2 below the x
		// index and 1 and 2 below the other indices.

		for (int j = (ind.xValue - 2); j <= (ind.xValue - 2); j++)
		for (int k = (ind.yValue - 2); k <= (ind.yValue - 1); k++)
		for (int l = (ind.theta - 2); l <= (ind.theta - 1); l++)
		{

			/*
			subMatrices[1][1][1].xValue += particle.xValue;
			subMatrices[1][1][1].yValue += particle.yValue;
			subMatrices[1][1][1].theta1 += sin(particle.theta);
			subMatrices[1][1][1].theta2 += cos(particle.theta);
			subMatrices[1][1][1].prob   += particle.prob;
			subMatrices[1][1][1].count++;
			*/

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 4
	//
	// In a lower boundary cell of the 10 x 10 x 10 for y index only

	if (ind.xValue > 1 && ind.xValue <= 9
		&& ind.yValue == 1
		&& ind.theta  > 1 && ind.theta <= 9)
	{
		// Add the particle to the submatrices with index 1 and 2 below
		// x and theta indices and 1 below y.

		for (int j = (ind.xValue - 2); j <= (ind.xValue - 1); j++)
		for (int k = (ind.yValue - 1); k <= (ind.yValue - 1); k++)
		for (int l = (ind.theta - 2); l <= (ind.theta - 1); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 5
	//
	// In a lower boundary cell of the 10 x 10 x 10 for x and y index

	if (ind.xValue == 1
		&& ind.yValue == 1
		&& ind.theta  > 1 && ind.theta <= 9)
	{
		// Add the particle to the submatrices with index 1 below the x
		// and y indices and 1 and 2 below the theta index.

		for (int j = (ind.xValue - 1); j <= (ind.xValue - 1); j++)
		for (int k = (ind.yValue - 1); k <= (ind.yValue - 1); k++)
		for (int l = (ind.theta - 2); l <= (ind.theta - 1); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 6
	//
	// In an upper boundary cell of the 10 x 10 x 10 for x index and lower
	// boundary for y index.

	if (ind.xValue == 10
		&& ind.yValue == 1
		&& ind.theta  > 1 && ind.theta <= 9)
	{
		// Add the particle to the submatrices with index 2 below the x
		// index, 1 below the y index and 1 and 2 below the theta index.

		for (int j = (ind.xValue - 2); j <= (ind.xValue - 2); j++)
		for (int k = (ind.yValue - 1); k <= (ind.yValue - 1); k++)
		for (int l = (ind.theta - 2); l <= (ind.theta - 1); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 7
	//
	// In an upper boundary cell of the 10 x 10 x 10 for y index only

	if (ind.xValue > 1 && ind.xValue <= 9
		&& ind.yValue == 10
		&& ind.theta  > 1 && ind.theta <= 9)
	{
		// Add the particle to the submatrices with index 1 and 2 below
		// x and theta indices and 2 below y.

		for (int j = (ind.xValue - 2); j <= (ind.xValue - 1); j++)
		for (int k = (ind.yValue - 2); k <= (ind.yValue - 2); k++)
		for (int l = (ind.theta - 2); l <= (ind.theta - 1); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 8
	//
	// In a lower boundary cell of the 10 x 10 x 10 for the x index and upper
	// boundary for the y index

	if (ind.xValue == 1
		&& ind.yValue == 10
		&& ind.theta  > 1 && ind.theta <= 9)
	{
		// Add the particle to the submatrices with index 1 below the x
		// index, 2 below the y, and 1 and 2 below the theta index.

		for (int j = (ind.xValue - 1); j <= (ind.xValue - 1); j++)
		for (int k = (ind.yValue - 2); k <= (ind.yValue - 2); k++)
		for (int l = (ind.theta - 2); l <= (ind.theta - 1); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 9
	//
	// In an upper boundary cell of the 10 x 10 x 10 for x and y indices.

	if (ind.xValue == 10
		&& ind.yValue == 10
		&& ind.theta  > 1 && ind.theta <= 9)
	{
		// Add the particle to the submatrices with index 2 below the x
		// index, 1 below the y index and 1 and 2 below the theta index.

		for (int j = (ind.xValue - 2); j <= (ind.xValue - 2); j++)
		for (int k = (ind.yValue - 2); k <= (ind.yValue - 2); k++)
		for (int l = (ind.theta - 2); l <= (ind.theta - 1); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 10
	//
	// Not in a boundary cell of the 10 x 10 x 10 for x or y, but lower
	// boundary for theta.

	if (ind.xValue > 1 && ind.xValue <= 9
		&& ind.yValue >  1 && ind.yValue <= 9
		&& ind.theta == 1)
	{
		// Add the particle to the submatrices with index 1 and 2 below
		// x and y and 1 below theta.

		for (int j = (ind.xValue - 2); j <= (ind.xValue - 1); j++)
		for (int k = (ind.yValue - 2); k <= (ind.yValue - 1); k++)
		for (int l = (ind.theta - 1); l <= (ind.theta - 1); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 11
	//
	// In a lower boundary cell of the 10 x 10 x 10 for x and theta only

	if (ind.xValue == 1
		&& ind.yValue >  1 && ind.yValue <= 9
		&& ind.theta == 1)
	{
		// Add the particle to the submatrices with index 1 below the x
		// and theta index and 1 and 2 below the y.

		for (int j = (ind.xValue - 1); j <= (ind.xValue - 1); j++)
		for (int k = (ind.yValue - 2); k <= (ind.yValue - 1); k++)
		for (int l = (ind.theta - 1); l <= (ind.theta - 1); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 12
	//
	// In an upper boundary cell of the 10 x 10 x 10 for x index and lower
	// for theta.

	if (ind.xValue == 10
		&& ind.yValue >  1 && ind.yValue <= 9
		&& ind.theta == 1)
	{
		// Add the particle to the submatrices with index 2 below the x
		// index, 1 below the theta and 1 and 2 below the y.

		for (int j = (ind.xValue - 2); j <= (ind.xValue - 2); j++)
		for (int k = (ind.yValue - 2); k <= (ind.yValue - 1); k++)
		for (int l = (ind.theta - 1); l <= (ind.theta - 1); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 13
	//
	// In a lower boundary cell of the 10 x 10 x 10 for y and theta index only

	if (ind.xValue > 1 && ind.xValue <= 9
		&& ind.yValue == 1
		&& ind.theta == 1)
	{
		// Add the particle to the submatrices with index 1 and 2 below
		// x and 1 below y and theta.

		for (int j = (ind.xValue - 2); j <= (ind.xValue - 1); j++)
		for (int k = (ind.yValue - 1); k <= (ind.yValue - 1); k++)
		for (int l = (ind.theta - 1); l <= (ind.theta - 1); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 14
	//
	// In a lower boundary cell of the 10 x 10 x 10 for x, y and theta index

	if (ind.xValue == 1
		&& ind.yValue == 1
		&& ind.theta == 1)
	{
		// Add the particle to the submatrices with index 1 all indices

		for (int j = (ind.xValue - 1); j <= (ind.xValue - 1); j++)
		for (int k = (ind.yValue - 1); k <= (ind.yValue - 1); k++)
		for (int l = (ind.theta - 1); l <= (ind.theta - 1); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 15
	//
	// In an upper boundary cell of the 10 x 10 x 10 for x index and lower
	// boundary for y and theta index.

	if (ind.xValue == 10
		&& ind.yValue == 1
		&& ind.theta == 1)
	{
		// Add the particle to the submatrices with index 2 below the x
		// index, 1 below the y and theta index

		for (int j = (ind.xValue - 2); j <= (ind.xValue - 2); j++)
		for (int k = (ind.yValue - 1); k <= (ind.yValue - 1); k++)
		for (int l = (ind.theta - 1); l <= (ind.theta - 1); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 16
	//
	// In an upper boundary cell of the 10 x 10 x 10 for y index and
	// lower for theta only

	if (ind.xValue > 1 && ind.xValue <= 9
		&& ind.yValue == 10
		&& ind.theta == 1)
	{
		// Add the particle to the submatrices with index 1 and 2 below
		// x, 2 below y and 1 below theta.

		for (int j = (ind.xValue - 2); j <= (ind.xValue - 1); j++)
		for (int k = (ind.yValue - 2); k <= (ind.yValue - 2); k++)
		for (int l = (ind.theta - 1); l <= (ind.theta - 1); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 17
	//
	// In a lower boundary cell of the 10 x 10 x 10 for the x and theta
	// index and upper boundary for the y index

	if (ind.xValue == 1
		&& ind.yValue == 10
		&& ind.theta == 1)
	{
		// Add the particle to the submatrices with index 1 below the x
		// and theta indices and 2 below the y index.

		for (int j = (ind.xValue - 1); j <= (ind.xValue - 1); j++)
		for (int k = (ind.yValue - 2); k <= (ind.yValue - 2); k++)
		for (int l = (ind.theta - 1); l <= (ind.theta - 1); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 18
	//
	// In an upper boundary cell of the 10 x 10 x 10 for x and y indices
	// and lower for theta .

	if (ind.xValue == 10
		&& ind.yValue == 10
		&& ind.theta == 1)
	{
		// Add the particle to the submatrices with index 2 below the x
		// and y index and 1 below the theta index.

		for (int j = (ind.xValue - 2); j <= (ind.xValue - 2); j++)
		for (int k = (ind.yValue - 2); k <= (ind.yValue - 2); k++)
		for (int l = (ind.theta - 1); l <= (ind.theta - 1); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 19
	//
	// Not in a boundary cell of the 10 x 10 x 10 for x or y, but upper
	// boundary for theta.

	if (ind.xValue > 1 && ind.xValue <= 9
		&& ind.yValue >  1 && ind.yValue <= 9
		&& ind.theta == 10)
	{
		// Add the particle to the submatrices with index 1 and 2 below
		// x and y and 2 below theta.

		for (int j = (ind.xValue - 2); j <= (ind.xValue - 1); j++)
		for (int k = (ind.yValue - 2); k <= (ind.yValue - 1); k++)
		for (int l = (ind.theta - 2); l <= (ind.theta - 2); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 20
	//
	// In a lower boundary cell of the 10 x 10 x 10 for x and upper for
	// theta only

	if (ind.xValue == 1
		&& ind.yValue >  1 && ind.yValue <= 9
		&& ind.theta == 10)
	{
		// Add the particle to the submatrices with index 1 below the x
		// 2 below theta index and 1 and 2 below the y.

		for (int j = (ind.xValue - 1); j <= (ind.xValue - 1); j++)
		for (int k = (ind.yValue - 2); k <= (ind.yValue - 1); k++)
		for (int l = (ind.theta - 1); l <= (ind.theta - 1); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 21
	//
	// In an upper boundary cell of the 10 x 10 x 10 for x and theta index

	if (ind.xValue == 10
		&& ind.yValue > 1 && ind.yValue <= 9
		&& ind.theta == 10)
	{
		// Add the particle to the submatrices with index 2 below the x
		// and theta index, and 1 and 2 below the y.

		for (int j = (ind.xValue - 2); j <= (ind.xValue - 2); j++)
		for (int k = (ind.yValue - 2); k <= (ind.yValue - 1); k++)
		for (int l = (ind.theta - 2); l <= (ind.theta - 2); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 22
	//
	// In a lower boundary cell of the 10 x 10 x 10 for y and upper for theta

	if (ind.xValue > 1 && ind.xValue <= 9
		&& ind.yValue == 1
		&& ind.theta == 10)
	{
		// Add the particle to the submatrices with index 1 and 2 below
		// x and 1 below y and 2 below theta.

		for (int j = (ind.xValue - 2); j <= (ind.xValue - 1); j++)
		for (int k = (ind.yValue - 1); k <= (ind.yValue - 1); k++)
		for (int l = (ind.theta - 2); l <= (ind.theta - 2); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 23
	//
	// In a lower boundary cell of the 10 x 10 x 10 for x, y and upper
	// for theta

	if (ind.xValue == 1
		&& ind.yValue == 1
		&& ind.theta == 10)
	{
		// Add the particle to the submatrices with index minus 1 for x
		// and y and -2 for theta.

		for (int j = (ind.xValue - 1); j <= (ind.xValue - 1); j++)
		for (int k = (ind.yValue - 1); k <= (ind.yValue - 1); k++)
		for (int l = (ind.theta - 2); l <= (ind.theta - 2); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 24
	//
	// In an upper boundary cell of the 10 x 10 x 10 for x and theta
	// index and lower boundary for y.

	if (ind.xValue == 10
		&& ind.yValue == 1
		&& ind.theta == 10)
	{
		// Add the particle to the submatrices with index 2 below the x
		// and theta index, and 1 below the y index

		for (int j = (ind.xValue - 2); j <= (ind.xValue - 2); j++)
		for (int k = (ind.yValue - 1); k <= (ind.yValue - 1); k++)
		for (int l = (ind.theta - 2); l <= (ind.theta - 2); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 25
	//
	// In an upper boundary cell of the 10 x 10 x 10 for y and theta index only

	if (ind.xValue > 1 && ind.xValue <= 9
		&& ind.yValue == 10
		&& ind.theta == 10)
	{
		// Add the particle to the submatrices with index 1 and 2 below
		// x, 2 below y and 2 below theta.

		for (int j = (ind.xValue - 2); j <= (ind.xValue - 1); j++)
		for (int k = (ind.yValue - 2); k <= (ind.yValue - 2); k++)
		for (int l = (ind.theta - 2); l <= (ind.theta - 2); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 26
	//
	// In a lower boundary cell of the 10 x 10 x 10 for the x and upper
	// boundary for the y and theta index

	if (ind.xValue == 1
		&& ind.yValue == 10
		&& ind.theta == 10)
	{
		// Add the particle to the submatrices with index 1 below the x
		// and 2 below the y and theta index.

		for (int j = (ind.xValue - 1); j <= (ind.xValue - 1); j++)
		for (int k = (ind.yValue - 2); k <= (ind.yValue - 2); k++)
		for (int l = (ind.theta - 2); l <= (ind.theta - 2); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}

	//
	// Case 27
	//
	// In an upper boundary cell of the 10 x 10 x 10 for x and y and
	// theta indices.

	if (ind.xValue == 10
		&& ind.yValue == 10
		&& ind.theta == 10)
	{
		// Add the particle to the submatrices with index 2 below all
		// indices

		for (int j = (ind.xValue - 2); j <= (ind.xValue - 2); j++)
		for (int k = (ind.yValue - 2); k <= (ind.yValue - 2); k++)
		for (int l = (ind.theta - 2); l <= (ind.theta - 2); l++)
		{

			//OSYSPRINT(("Adding to submatrix %d, %d, %d\n", j, k, l));

			subMatrices[j][k][l].xValue += particle.xValue;
			subMatrices[j][k][l].yValue += particle.yValue;
			subMatrices[j][k][l].theta1 += sin(particle.theta);
			subMatrices[j][k][l].theta2 += cos(particle.theta);
			subMatrices[j][k][l].prob += particle.prob;
			subMatrices[j][k][l].count++;
		}
	}
}
//-----------------------------------------------------------------------------
//
// findBestSubMatrix
//
// Find the best 2 x 2 x 2 submatrix, that is the one with the largest set
// of particles in it, average the location of these particles, and then
// set the value of mostLikelyLocation.
//

void Mcl::findBestSubMatrix(void)
{
	clusterBucket bestBucket;
	locIndex      bestBucketIndex;
	int           maxCount;

	//OSYSPRINT(("LO> Finding best submatrix...\n"));

	maxCount = 0;

	// Set these values to signal no best bucket found

	bestBucketIndex.xValue = -1;
	bestBucketIndex.yValue = -1;
	bestBucketIndex.theta = -1;

	// Just go through the sub-matrix list, looking at the count. The best in
	// this sense is the one with the most particles in it and thus the highest 
	// count.
	//
	// Note that we only look through places 0-8, and ignore 9; that is because
	// I don't really understand why we need the ninth place, but the code 
	// crashes without it.

	for (int j = 0; j <= 8; j++)
	for (int k = 0; k <= 8; k++)
	for (int l = 0; l <= 8; l++)
	{
		//OSYSPRINT(("LO> Matrix %d, %d, %d has count %d\n", 
		//	     j, k, l, subMatrices[j][k][l].count));
		if (subMatrices[j][k][l].count > maxCount)
		{
			maxCount = subMatrices[j][k][l].count;
			bestBucketIndex.xValue = j;
			bestBucketIndex.yValue = k;
			bestBucketIndex.theta = l;
		}
	}

	/*
	OSYSPRINT(("LO> Best submatrix is %d, %d, %d\n",
	bestBucketIndex.xValue,
	bestBucketIndex.yValue,
	bestBucketIndex.theta));
	*/

	// If we have a reasonable best bucket, average it, otherwise 
	// return a random location

	if (bestBucketIndex.xValue != -1)
	{
		mostLikelyLocation
			= averageLocation(subMatrices[bestBucketIndex.xValue]
			[bestBucketIndex.yValue]
		[bestBucketIndex.theta]);
	}
	else
	{
		mostLikelyLocation.xValue = (float)drand48() * MAX_X;
		mostLikelyLocation.yValue = (float)drand48() * MAX_Y;
		mostLikelyLocation.theta = (float)drand48() * TWO_PI;
		mostLikelyLocation.prob = (double)1 / PARTICLES;
	}
}
loc Mcl::averageLocation(clusterBucket aBucket)
{

	//double prob   = 0;
	loc    averageLocation;

	//OSYSPRINT(("LO> Averaging location values\n"));

	/*
	OSYSPRINT(("LO> Bucket values are %d, %d, %f, %f, %d\n",
	(int)aBucket.xValue,
	(int)aBucket.yValue,
	aBucket.theta1,
	aBucket.theta2,
	aBucket.count));
	*/

	// Unlike the German Team, we take the probability of the average to be the
	// sum of all the probabilities of all the particles in the bucket/cluster.
	// 
	// The GT way of doing things leads me to believe that their particle 
	// probabilities are rather different to ours...

	if (aBucket.count > 0)
	{
		averageLocation.xValue = aBucket.xValue / aBucket.count;
		averageLocation.yValue = aBucket.yValue / aBucket.count;

		// We have to worry about the angle since we measure angles
		// between 0 and 2PI while atan returns a value between PI and
		// -PI... so add 2PI to negative values.

		averageLocation.theta = (float)atan2(aBucket.theta1, aBucket.theta2);

		if (averageLocation.theta < 0)
		{
			averageLocation.theta += (float)TWO_PI;
		}

		averageLocation.prob = aBucket.prob;
	}
	else
	{
		averageLocation.xValue = (float)drand48() * MAX_X;
		averageLocation.yValue = (float)drand48() * MAX_Y;
		averageLocation.theta = (float)drand48() * TWO_PI;
		averageLocation.prob = (double)1 / PARTICLES;
	}

	return averageLocation;
}
// End of additions for German/UTA-style localization
//
//=============================================================================
//-----------------------------------------------------------------------------
//
// resample 
//
// Resample from the set of particles in the array indicated by aPitch. The
// resampled values are placed back into the original at the end of the 
// sampling.
//
// According to Thrun et al. p109, this may be problematic. The solution is 
// to use the low variance sampler above.


void Mcl::resample(loc* aPitch, double totalProb)
{
	loc    otherPitchCopy[PARTICLES];

	// First build an array so that every element is sampled from aPitch...

	for (int i = 0; i<PARTICLES; i++)
	{
		otherPitchCopy[i] = aPitch[randomSample(aPitch, totalProb)];
	}

	// ... then copy those elements back into aPitch.

	for (int i = 0; i<PARTICLES; i++)
	{
		aPitch[i] = otherPitchCopy[i];
	}
}

//-----------------------------------------------------------------------------
//
// gaussianWithMean
//
// A univariate gaussian p(x) with mean. input x, mean m and stdev s

double Mcl::gaussianWithMean(float x, float mx, float s)
{

	double f, e, p;

	f = 1 / (sqrt(2 * M_PI)*s);
	e = -0.5 * pow(((x - mx) / s), 2);
	p = f*exp(e);

	return p;
}
//-----------------------------------------------------------------------------
//
// gaussian
//
// A univariate gaussian distribution p(x), input is x and the stdev is s.
//

double Mcl::gaussian(float x, float s)
{

	double f, e, p;

	f = 1 / (sqrt(2 * M_PI)*s);
	e = -0.5 * pow((x / s), 2);
	p = f*exp(e);

	return p;
}
//-----------------------------------------------------------------------------
//
// myCdfGaussianPinv
//
// Given the stdev of a gaussian and a probability prob, tell us the
// offset from the mean that gives a cumulative probability that is at
// least as big as prob.

float Mcl::myCdfGaussianPinv(double prob, float stdev, float increment)
{

	float  limit, offset, index;
	double cumulative;

	// 0.9973 is the area under 6 standard deviations around the
	// mean. So if prob is less than 0.00135, we assume that offset is
	// 3 of those stdevs.

	limit = (3 * stdev);

	if (prob <= 0.00135)
	{
		offset = -1 * limit;
	}
	else
	{
		cumulative = 0.00135;
		index = (-1 * limit);

		// Make steps of increment mm, assuming that the area under the
		// curve between x and x + increment is:
		//
		//   increment.Prob(x + increment) 
		//
		// which, of course, is only exact for infinitesimal increment, and
		// will overestimate the area on the left of the mean and underestimate
		// area on the right.

		while (cumulative < prob &&  index < limit)
		{
			cumulative += gaussian(index, stdev) * increment;
			index += increment;
		}

		/*
		OSYSPRINT(("LO> cumulative %5f\n",cumulative));
		OSYSPRINT(("LO> index %5f\n",index));
		*/

		// If we left the loop because we exceeded the 3stdev upper limit
		// cap the offset at 3 stdevs. Otherwise, our index is the offset we
		// want.

		if (index >= limit)
		{
			offset = limit;
		}
		else
		{
			offset = index;
		}
	}

	return offset;
}
//-----------------------------------------------------------------------------
//
// notInTop10
//
// Function used to help preserve the particles with the current
// highest probability. 
//
// Since we have to search through the full set of preserved particles
// for each particle in turn, this is not terribly efficient.
//
// If we are doing the check, findTop10 is called after updating to
// build an ordered list of the LOCS_SORTED (the name is a hangover
// from an earlier version of localization) most probable particles,
// and then notInTop10 returns false if its argument is not in the top
// LOC_PRESERVED of these.
//
// Clearly we must have LOCS_PRESERVED <= LOCS_SORTED

bool Mcl::notInTop10(int index)
{

	if (PRESERVE_PARTICLES)
	{
		for (int i = 0; i<LOCS_PRESERVED; i++)
		{
			if (top10Index[i] == index)
			{
				return false;
			}
		}
	}

	return true;
}



//-----------------------------------------------------------------------------
//
// initialize
//
// Called by Localization.cc to initialize the mcl instance of Mcl that is
// created to perform localization.
//

void Mcl::initialize()

{

	// Set random number stuff

	long int seed = 12341;
	srand(seed);
	srand48(seed);

	// Set up global datastructures

	initializePitch();

}
int main(int argc, char** argv)
{
	int cam_count;

	//仅仅获取摄像头数目		Get the Camera number
	cam_count = CCameraDS::CameraCount();
	printf("There are %d cameras.\n", cam_count);


	//获取所有摄像头的名称	Capture all the name of the Camera(if it has one)
	for (int i = 0; i < cam_count; i++)
	{
		char camera_name[1024];
		int retval = CCameraDS::CameraName(i, camera_name, sizeof(camera_name));

		if (retval >0)
			printf("Camera #%d's Name is '%s'.\n", i, camera_name);
		else
			printf("Can not get Camera #%d's name.\n", i);
	}

	if (cam_count == 0)
		return -1;

	// 创建2个摄像头类 Declare/Define two Camera Class
	CCameraDS camera1;
	CCameraDS camera2;

	//打开第一个摄像头	Open the First Camera
	//if(! camera.OpenCamera(0, true)) //弹出属性选择窗口
	if (!camera1.OpenCamera(0, false, 320, 240)) //不弹出属性选择窗口，用代码制定图像宽和高	 Define the image width and height
	{
		fprintf(stderr, "Can not open camera./n");
		return -1;
	}
	//打开第二个摄像头		Open the second Camera
	camera2.OpenCamera(2, false, 320, 240);


	cvNamedWindow("Multiple Cameras");

	// 初始化在子图像中显示字符的字体格式
	CvFont tFont;
	cvInitFont(&tFont, CV_FONT_HERSHEY_COMPLEX, 0.5f, 0.7f, 0, 1, 8);

	char cam1str[] = "Camera #1";
	char cam2str[] = "Camera #2";

	// 为读取系统时间信息分配内存	storage allocation
	char timestr[25];
	memset(timestr, 0, 25 * sizeof(char));

	while (1)
	{
		//获取一帧	Capture a single frame
		IplImage *pFrame1 = camera1.QueryFrame();
		IplImage *pFrame2 = camera2.QueryFrame();

		// 获取当前帧的灰度图		Get the Grayscale image of the current frame
		IplImage* frame_gray_1 = cvCreateImage(cvGetSize(pFrame1), pFrame1->depth, 1);
		IplImage* frame_gray_2 = cvCreateImage(cvGetSize(pFrame2), pFrame2->depth, 1);
		cvCvtColor(pFrame1, frame_gray_1, CV_RGB2GRAY);
		cvCvtColor(pFrame2, frame_gray_2, CV_RGB2GRAY);

		// 对灰度图像进行Canny边缘检测				Do the Canny edge detection with the Grayscale image	 
		// 然后将图像通道数改为三通道				Then change the channel into 3 channel
		IplImage* frame_canny_1 = cvCreateImage(cvGetSize(pFrame1), pFrame1->depth, 1);
		IplImage* frame_canny_2 = cvCreateImage(cvGetSize(pFrame2), pFrame2->depth, 1);
		IplImage* frame1 = cvCreateImage(cvGetSize(pFrame1), pFrame1->depth, pFrame1->nChannels);
		IplImage* frame2 = cvCreateImage(cvGetSize(pFrame2), pFrame2->depth, pFrame2->nChannels);
		cvCanny(frame_gray_1, frame_canny_1, 120, 300, 3);
		cvCanny(frame_gray_2, frame_canny_2, 120, 300, 3);
		Mat midImage, dstImage;		
		Mat midImage(frame_canny_1);
		cvtColor(midImage, dstImage, CV_GRAY2BGR);
		    // 检测直线，最小投票为90，线条不短于50，间隙不小于10
		vector<Vec4i> lines;
		HoughLinesP(midImage, lines, 1, CV_PI / 180, 80, 50, 10);
		for (size_t i = 0; i < lines.size(); i++)
		{
			Vec4i l = lines[i];	
			line(dstImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(186, 88, 255), 1, CV_AA);
		}
		namedWindow("Lines");
		imshow("LINES", dstImage);
		
		    




		cvCvtColor(frame_canny_1, frame1, CV_GRAY2BGR);
		cvCvtColor(frame_canny_2, frame2, CV_GRAY2BGR);
		

		//-------------------------------------------------------------------------------------------------




		class  Mcl		initialize();
		class  Mcl	initializePitch();
		class  Mcl   update(motionData*, int, int, int, float, float);


		//--------------------------------------------------------------------------


		// 获取系统时间信息 get the system time info
		time_t rawtime;
		struct tm* timeinfo;

		rawtime = time(NULL);
		timeinfo = localtime(&rawtime);
		char* p = asctime(timeinfo);

		// 字符串 p 的第25个字符是换行符 '\n'	the 25th character in the string p is "\n"
		// 但在子图像中将乱码显示				which was shown in error codes
		// 故仅读取 p 的前 24 个字符			So only show the first 24 characters
		for (int i = 0; i < 24; i++)
		{
			timestr[i] = *p;
			p++;
		}
		p = NULL;

		// 在每个子图像上显示摄像头序号以及系统时间信息			title the each sub-video camera in number and system time info
		cvPutText(pFrame1, cam1str, cvPoint(95, 15), &tFont, CV_RGB(255, 0, 0));
		cvPutText(pFrame2, cam2str, cvPoint(95, 15), &tFont, CV_RGB(255, 0, 0));
		cvPutText(frame1, cam1str, cvPoint(95, 15), &tFont, CV_RGB(255, 0, 0));
		cvPutText(frame2, cam2str, cvPoint(95, 15), &tFont, CV_RGB(255, 0, 0));

		cvPutText(pFrame1, timestr, cvPoint(5, 225), &tFont, CV_RGB(255, 0, 0));
		cvPutText(pFrame2, timestr, cvPoint(5, 225), &tFont, CV_RGB(255, 0, 0));
		cvPutText(frame1, timestr, cvPoint(5, 225), &tFont, CV_RGB(255, 0, 0));
		cvPutText(frame2, timestr, cvPoint(5, 225), &tFont, CV_RGB(255, 0, 0));

		// 显示实时的摄像头视频		show the direct multiple images
		cvShowMultiImages("Multiple Cameras", 4, pFrame1, pFrame2, frame1, frame2);


		//cvWaitKey(33);
		int key = cvWaitKey(33);
		if (key == 27) break;

		cvReleaseImage(&frame1);
		cvReleaseImage(&frame2);
		cvReleaseImage(&frame_gray_1);
		cvReleaseImage(&frame_gray_2);
		cvReleaseImage(&frame_canny_1);
		cvReleaseImage(&frame_canny_2);
	}

	camera1.CloseCamera(); //可不调用此函数，CCameraDS析构时会自动关闭摄像头
	camera2.CloseCamera();

	cvDestroyWindow("Multiple Cameras");

	return 0;
}
