#ifndef __Utility__
#define __Utility__

#ifdef __APPLE__
        #include <sys/uio.h>
#else
        #include <sys/io.h>
#endif

#include <iostream>
// #include <opencv2/tracking.hpp>
// #include <opencv2/core/core.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <tiffio.h>
// namespace libtiff {
//     #include "tiffio.h"
// }

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


struct MeshGrid_struct {
  cv::Mat X, Y;
};



// global Parameters
cv::Mat discArray(vector<double> shape, double radius, vector<double> origin);
MeshGrid_struct meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y);

// int player();





#endif //