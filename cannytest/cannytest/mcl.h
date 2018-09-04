//----------------------------------------------------------------------------
//
// mcl.h
//
//  Header File for Localization Module when adopting the particle filtering
//  approach
//
//----------------------------------------------------------------------------

//#include <stack>                         // Provide C++ STL stack definition
                                         // used to cluster particles to 
using namespace std;                     // figure out where we are.

//----------------------------------------------------------------------------
//
// Preprocessor stuff
//

// General things

#define PARTICLES 200            // The number of samples we use to
				 // represent the pitch.

//Old pitch

#define MAX_X 6000               // Limits on the x and y coordinates. Define
#define MIN_X 0                  // both to make easy reconfiguration for
#define MAX_Y 8000               // partial pitches.
#define MIN_Y 0

/*

// New pitch

#define MAX_X 5400               // Limits on the x and y coordinates. Define
#define MIN_X 0                  // both to make easy reconfiguration for
#define MAX_Y 3650               // partial pitches.
#define MIN_Y 0

*/

#define TWO_PI 6.2831853071796   // Since we use these so often...
#define MINUS_TWO_PI -6.2831853071796
#define HALF_PI 1.570796327
#define M_PI 3.14159265359

// Motion model

#define X_MOTION_STDEV  0.1      // Proportions of the distance moved that
#define Y_MOTION_STDEV  0.1      // we take as the standard deviation
#define A_MOTION_STDEV  0.2      // in the motion error distribution.

#define X_ZERO_MOTION_STDEV 10   // Absolute values we take as standard 
#define Y_ZERO_MOTION_STDEV 10   // deviations when we haven't moved.
#define A_ZERO_MOTION_STDEV 0.1

//#define INCREMENT_X 0.1         // Proportion of the stdev that is used as
//#define INCREMENT_Y 0.1	  // a stepsize for the approximate cumulative 
//#define INCREMENT_A 0.1         // gaussian calculations.

// Sensor model

#define R_SENSOR_STDEV 0.15      // The variance of radius and angle in the
#define T_SENSOR_STDEV 0.52      // sensor model. The angle stdev is constant,
                                 // the distance stdev is a proportion. For
                                 // distance 0.15 is the UWash suggestion.

// Adding new samples in the particle filter

#define LOCS_SORTED 30           // FindTop10 locates the most probable
                                 // LOCS_SORTED particles.

#define LOCS_PRESERVED 30        // The number of samples (from the most likely
                                 // down) that we won't overwrite when adding
                                 // random samples in the particle filter.
                                 // must be <= LOCS_SORTED.

#define PRESERVE_PARTICLES false // Should we preserve these particles?

#define RAND_PROPORTION 0.05     // The proportion of new random samples
                                 // added each time we update with an
                                 // observation under the default scheme.

#define P_THRESHOLD 0.00025      // Threshold for sensor resetting,
				 // taken from Fox and Gutmann.

#define ETA_LONG  0.001          // Values for sensor resetting plus
#define ETA_SHORT 0.1            // taken from Fox and Gutmann.
#define NU        2

// Debugging

#define LOCS_DISPLAYED PARTICLES // How many particles (unsorted) we print
                                 // put in printPitchUnsort

//-----------------------------------------------------------------------------
//
// Data types
//
  
// We now represent the "pitch" as an array of PARTICLES
// particles/samples, each of which has an x, y, theta value and a
// probability.

// Each location is a structure

struct location
{
  float  xValue;       // x coordinate
  float  yValue;       // y coordinate
  float  theta;        // angle from north
  double prob;         // probability of this location
};

// These are *global* coordinates.
//
// x and y are in mm from the corner to the left of the yellow goal (the 
// southwest corner, thinking in compass directions). theta is measured
// from north in a clockwise direction from 0 (north) through pi/2 (east) and  
// pi (south) and negatively from 0 through -pi/2 (west) to -pi (south).

// We have a typedef for convenience

typedef location loc;

// When we cluster particles, we need to create a matrix in which each
// element is a summary of a set of locs and a count of the number of locs that
// fall in that bucket.

struct clusterBucket
{
  float       xValue;      // The sum of values across elements in the bucket
  float       yValue;      // We need two angle components to average angles
  float       theta1;      // correctly. 
  float       theta2;      //
  int         count;       // The number of elements in the bucket
  float       prob;        // Summed probability of all the samples in the
                           // bucket.
};

// We also want to carry around some triples of x, y and angle, so let's have:

struct locationData
{
  float  xValue;        // x component
  float  yValue;        // y component
  float  theta;         // angle component 
};

typedef locationData locData;

// And we need the same kind of thing but for polar coordinates

struct polarData
{
  float  radius;         // radius component
  float theta;           // angle component
};

typedef polarData polData;

// At times we will need to deal with 3D arrays of location data, and
// will use the following to group indices for such arrays.

struct locationIndex
{
  int  xValue;        // x component
  int  yValue;        // y component
  int  theta;         // angle component 
};
  
typedef locationIndex locIndex;

// Finally, we need a motion datastructure that includes the same kind
// of data, but also timestamps.

struct motionData
{
  float  xValue;        // x component as a fraction of MAX_DX
  float  yValue;        // y component as a fraction of MAX_DY
  float  theta;         // angle component as a fraction of MAX_DA
  double time;          // timestamp (in milliseconds)
};

// Places where we want to switch between different options:

enum motionTypes {JITTER, NO_JITTER};

enum sampleTypes {SENSOR_RESETTING, SENSOR_RESETTING_PLUS, DEFAULT};

enum searchTypes {GERMAN, MAXPROB, MAXPROB_PLUS};

//-----------------------------------------------------------------------------
//
// Class signature
//

class Mcl{

   public:

   //
   // Attributes
   //

   // The pitch is represented by an array of locations.

   location pitch[PARTICLES];
  
   //
   // Methods
   //

    Mcl(){};
    ~Mcl(){};

    void          initializePitch         (void);
    void          update                  (motionData*, int, int, int, float, float);
    void          updateWithMotion        (motionData);
    void          updateWithObservation   (int, float, float);  
    void          update                  (motionData, int, float, float);
    int           randomSample            (loc*, double);
    int           pickSample              (double, loc*);
    double        totalProb               (loc*);
    double        maxProb                 (loc*);
    loc*          normalize               (loc*, double);
    loc           kinematics              (loc, motionData);
    locData       pickRandomMotion        (motionData);
    locData       changeFrameLocalToGlobal(locData, loc);
    float         compensateForFullTurn   (float);
    float         checkXBounds            (float);
    float         checkYBounds            (float);
    double        getTime                 (void);
    double        computeWeight           (loc, int, float, float);
    void          resample                (loc*, double);
    void          lowVarianceResample     (loc*, double);
    void          addRandomElementsTo     (loc*, double);
    polData       computeDistance         (loc, int);
    float         adjustForCorrectMean    (float);
    double        gaussian2D              (polData, polData, polData);
    float         myCdfGaussianPinv       (double, float, float);
    float         sampleFromGaussian      (float);
    double        gaussian                (float, float);
    double        gaussianWithMean        (float, float, float);
    double        probabilityFilter       (double, double); 
    int           howManySamples          (loc*, double);
    double        max                     (double, double);
    void          probabilisticSearch     (loc*, double);
    loc           whereAmI                (void);
    loc*          findTop10               (void);
    bool          notInTop10              (int);
    loc           whereAmIReally          (void);
    void          findMostLikelyLocation  (void);
    void          clusterParticles        (loc*);
    void          initialiseMatrices      (void);
    void          addToSubMatrix          (loc, locIndex);
    void          findBestSubMatrix       (void);
    loc           averageLocation         (clusterBucket);
    void          printPitchUnsort        (loc*);
    void          printPitchSort          (loc*);
    void          printAPitch             (void);
    void          close                   (void);
    void          initialize              (void);
    loc           returnMostLikelyLocation(void);
    loc           printMostLikelyLocation (void);

};

