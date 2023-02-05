#ifndef __Utility__
#define __Utility__

#ifdef __APPLE__
        #include <sys/uio.h>
#else
        #include <sys/io.h>
#endif

#include <iostream>

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iterator>
#include <stdio.h>
#include <fstream>
#include <string>
#include <sstream>
#include <typeinfo>
#include <vector>
#include <cmath> //math lib
#include <chrono> //for tic toc



using namespace std;
using namespace cv;
// using namespace libtiff;



// global Parameters
cv::Mat discArray(vector<double> shape, double radius, vector<double> origin = {-1,-1});




#endif //