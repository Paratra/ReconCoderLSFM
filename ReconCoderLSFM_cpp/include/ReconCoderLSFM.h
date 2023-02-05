#ifndef __ReconCoderLSFM__
#define __ReconCoderLSFM__

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
// using namespace libtiff;

// global Parameters
// vector<string> _img_list; // name of all images.
// const char * _corr_file_path;
string image_data_path;
// string _local_data_path;

// libtiff::TIFF* tif; // not using 

int numangle, numphase, size;
Size_<int> WIDTH, LENGTH;
double wl, na, pixel_size, angle1, angle2, angle3, patternspacing1, patternspacing2, patternspacing3;
double angle1_deg, angle2_deg, angle3_deg, space1_pix, space2_pix, space3_pix;


struct psf_params{
	double wl=-1;
	double na=-1;
	double dx=-1;
	double nx=-1;
};


// class MyClass {
// 	public:
//     // Parameterized constructor
//     MyClass(int a, int b) {
//         x = a;
//         y = b;
//     }

//     // Member function to get the value of x
//     int getX() const {
//         return x;
//     }


// 	private:
// 	int x,y;
	
// };



class PSF
{
private:
	// parameters 


public:
	double wl, na, n2, dx,nx,dp, radius;
	vector<double> zarr;
	int zarr_num;
	cv::Mat bpp;

	PSF();
	void setParams(psf_params new_params);

};







class RECON_2D
{
public: //Accessable for all entities.

	// Parameterized constructor
    RECON_2D(vector<cv::Mat> input_img_stack, int input_nangles, int input_nphs, double input_wavelength, double input_na, double input_dx, double input_mu, double input_cut_off) ;


//Parameters
	


//Functions

	// void read_corr(const char *path);
	// void show_img(string local_data_path, string img_name, string image_col, string image_row);
    // stringstream get_filename_from_date_cameranNum(string img_date, string img_camera_id);

    vector<cv::Mat> getallangleield(bool verbose);
	vector<vector<cv::Mat>> subback(vector<cv::Mat>);
	cv::Mat getpsf();



private: //Only accessable for this class members.

	// Parameters
	vector<cv::Mat> img_stack;
	int nang, nph, nx, ny;
	double wl, na, dx, mu, cut_off, radius;
	vector<vector<cv::Mat>> img;
	cv::Mat psf;


	// vector<const char*>_content;
	// vector<file_info>_corr_info;

	// stringstream file_name;
	// stringstream temp_name;

	// Mat _current_img;

	// file_info _file_info;
	// string str;
	//
	// functions ??
	// void find_img_and_copy(string img_date, string img_camera_id);
    // void sear_coordinates(vector<file_info>corr_info);

};


#endif //