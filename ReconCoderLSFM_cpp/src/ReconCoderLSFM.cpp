// COPYRIGHT: Ming Song UGA Kner Lab 2023
// __function__: SIM 2D reconstruction
// __author__:  MingSong

#include "ReconCoderLSFM.h"
#include "Utility.h"
// ############################################################

bool is_file_exist(string fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

template<typename T>
std::vector<T> arange(T start, T stop, T step = 1) {
    std::vector<T> values;
    for (T value = start; value < stop; value += step)
        values.push_back(value);
    return values;
}

template<typename T>
void print_vector(vector<T> V){
    for (size_t i = 0; i < V.size(); i++)
    {
        cout << V[i] << endl;
    }
}

bool visualize_vector_of_img(vector<cv::Mat> images, double max_value, int duration){
    ///////// visualize data
    for (int i = 0; i < images.size(); i++) {
        cv::Mat img = images[i];
        Mat ff_show;
        img.convertTo( ff_show, CV_32FC1, 1.0/max_value);  // [0..1] range
        imshow("TIF Image", ff_show);
        waitKey(duration);
    }
    return 1;
}


PSF::PSF(){
    wl = 0.525;
	na = 0.8;
	n2 = 1.512;
	dx = 0.135;
	nx = 256;
	dp = 1/(nx*dx);
	radius = (na/wl)/dp;
    zarr_num = 15;
	for (size_t i = 0; i < zarr_num; i++)
    {
        zarr.push_back(0);
    }
}

void PSF::setParams(psf_params new_params){
    if (new_params.wl!=-1)
    {
        this->wl = new_params.wl;
    }
    if (new_params.na!=-1)
    {
        this->na = new_params.na;
    }
    if (new_params.dx!=-1)
    {
        this->dx = new_params.dx;
    }
    if (new_params.nx!=-1)
    {
        this->nx = new_params.nx;
    }
    this->dp = 1/(nx*dx);
	this->radius = (na/wl)/dp;

}


cv::Mat PSF::getFlatWF(){
    
    vector<double> size_vec{this->nx, this->nx};
    vector<double> origin{-1, -1};

    this->bpp = discArray(size_vec, this->radius, origin);

    return this->bpp;
}




RECON_2D::RECON_2D(vector<cv::Mat> input_img_stack, int input_nangles, int input_nphs, double input_wavelength, double input_na, double input_dx, double input_mu, double input_cut_off) {
    img_stack = input_img_stack;
    nx = input_img_stack[0].rows;
    ny = input_img_stack[0].cols;
    nang = input_nangles;
    nph = input_nphs;
    wl = input_wavelength;
    na = input_na;
    dx = input_dx;
    mu = input_mu;
    cut_off = input_cut_off;
    img = subback(input_img_stack);
    // do not need to do self.img = self.img.reshape(self.nang,self.nph,nx,ny), the result is already reshaped from suback
    psf = getpsf();
}


cv::Mat RECON_2D::getpsf(){
    double dx = this->dx / 2;
    double nx = this->nx * 2;

    PSF psf;
    psf_params new_psf_params;
    new_psf_params.wl = this->wl;
    new_psf_params.na = this->na;
    new_psf_params.dx = dx;
    new_psf_params.nx = nx;

    psf.setParams(new_psf_params);
    this->radius = psf.radius;

    psf.bpp = psf.getFlatWF();


    cv::Mat wf = psf.bpp;

    cv::Mat_<cv::Vec2d> input = (cv::Mat_<cv::Vec2d>(5, 5) << 1, 0, 2, 0, 3,
                                                              0, 0, 0, 0, 0,
                                                              4, 0, 5, 0, 6,
                                                              0, 0, 0, 0, 0,
                                                              7, 0, 8, 0, 9);

    // Create an output matrix for the Fourier coefficients
    cv::Mat_<cv::Vec2d> output;

    // Compute the forward DFT
    cv::dft(input, output, cv::DFT_COMPLEX_OUTPUT);

    // Compute the inverse DFT
    cv::Mat_<cv::Vec2d> inverse;
    cv::dft(output, inverse, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);

    // Print the input and output matrices
    std::cout << "Input:\n" << input << std::endl;
    std::cout << "Output:\n" << output << std::endl;
    std::cout << "Inverse:\n" << inverse << std::endl;

    // cv::Mat fft_output;
    // cv::dft(wf, fft_output, cv::DFT_COMPLEX_OUTPUT);

    // cv::Mat fft_output_abs = cv::abs(fft_output);

    // cv::Mat fft_output_abs_squared;
    // cv::pow(fft_output_abs, 2, fft_output_abs_squared);

    // cout << fft_output_abs_squared(cv::Range(0,5),cv::Range(0,5)) << endl;


    // cout << fft_output_abs_squared << endl;
    // cv::imshow("window",abs_fft_output_squared);
    // cv::waitKey(0);
    abort();



    // int dd = player();
    // cout << vect[0] << endl;
    
    abort();

    cv::Mat aaa;
    return aaa;
}





vector<vector<cv::Mat>> RECON_2D::subback(vector<cv::Mat> img_stack){
    vector<cv::Mat> img = img_stack;
    vector<vector<cv::Mat>> result_img = {};
    int nz = img.size();

    // for (size_t i = 0; i < nz; i++)

    size_t i = 0;
    for (size_t n = 0; n < this->nang; n++)
    {
        vector<cv::Mat> temp_img = {};
        for(size_t m = 0; m < this->nph; m++)
        {
            cv::Mat data = img[i].clone();
            cv::Mat cur_result = img[i].clone();
            cv::Mat hist;

            // find min max value
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(data, &minVal, &maxVal, &minLoc, &maxLoc);

            int histSize = maxVal - minVal;

            float range[] = { (float)(minVal), (float)(maxVal) }; //the upper boundary is exclusive
            const float* histRange[] = { range };
            cv::calcHist(&data, 1, 0, cv::Mat(), hist, 1, &histSize, histRange);

            vector<double> bin_edges = arange<double>(minVal, maxVal);

            // find min max value of that hist
            double minVal_hist, maxVal_hist;
            cv::Point minLoc_hist, maxLoc_hist;
            cv::minMaxLoc(hist, &minVal_hist, &maxVal_hist, &minLoc_hist, &maxLoc_hist);  

            // maxLoc_hist.y is the index, .x always 0
            double bg = bin_edges[maxLoc_hist.y+1];
            // thresh all below bg to zeros
            // cout <<data[0,0] <<endl;
            // abort();
            // cv::threshold(data, data, bg, maxVal, cv::THRESH_TOZERO);
            for (size_t j = 0; j < data.rows; j++)
            {
                for (size_t k = 0; k < data.cols; k++)
                {
                    
                    if (cur_result.at<float>(k, j) <= bg)
                    {
                        cur_result.at<float>(k, j) = 0;
                    }
                    else
                    {
                        // cout << cur_result.at<float>(k, j) - bg << endl;
                        cur_result.at<float>(k, j) = cur_result.at<float>(k, j) - bg;
                    }
                }
            }

            temp_img.push_back(cur_result);
            // cout << temp_img.size() << endl;
            // result_img.push_back(cur_result);
            i++;
        }
        result_img.push_back(temp_img);
    }
    
    // // cout << result_img.size() <<endl;
    // visualize_vector_of_img(result_img, 255, 0);
    // abort();

    return result_img;
}


vector<cv::Mat> RECON_2D::getallangleield(bool verbose){

    vector<cv::Mat> img = this->img_stack;
    int numangle = this->nang;
    int width = this-> nx;
    int length = this->ny;


    std::vector<cv::Mat> temp_vec;
    // while loop
    do{
        if (verbose)
        {
            cout << "size of img: " << img.size() << endl;
        }
        
    
        // init a matrix with all zero
        cv::Mat temp(width,length,CV_32F);
        temp.setTo(0);

        for (size_t i = 0; i < numangle; i++)
        {   
            temp += img[i];
        }
        
        temp_vec.push_back(temp); //append temp to temp vector
        img.erase(img.begin(), img.begin() + numangle);

    } while (img.size()>0);

    vector<cv::Mat> result_vec; 
    cv::Mat result_mat(width,length,CV_32F);
    result_mat.setTo(0);

    for (size_t i = 0; i < temp_vec.size(); i++)
    {
        result_mat += temp_vec[i];
        
        if ((i+1) % numangle == 0)
        {   
            result_mat = result_mat/numangle;
            result_vec.push_back(result_mat.clone());
            result_mat.setTo(0);
        }
        
    }

    return result_vec;
}



int main() {
    clock_t start = clock();




    // read img using opencv
    std::vector<cv::Mat> images;
    image_data_path = "../data/sim_si2d_1.tif";
    // imreadmulti(image_data_path, images, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
    // imreadmulti("../../../data/20230119/organoid_d7_cae_yg488_10um_500nmstep_9px_sin_20482048_40.tif",images, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
    imreadmulti("../data/sim_si2d_1.tif",images, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);


    if (images.size() == 0){
        std::cout << "Unable to open TIFF file." << std::endl;
        return -1;
    }else{
        WIDTH = images[0].size();
        LENGTH = images[1].size();
        assert(WIDTH == LENGTH);
    }
    
    // Parameters 

    numangle = 3;
    numphase = 3;
    wl = 0.525;
    na = 0.8;
    size = WIDTH.width;
    pixel_size = 0.135;
    angle1 = (-90.002+90) /360*2*M_PI ;
    angle2 = (29.989+90) /360*2*M_PI ;
    angle3 = (149.99+90) /360*2*M_PI ;
    patternspacing1 = 1/(52.656*(1/(size*pixel_size)));
    patternspacing2 = 1/(52.661*(1/(size*pixel_size)));
    patternspacing3 = 1/(52.673*(1/(size*pixel_size)));



    cout << "Angle1 Init angle and spacing: (" << angle1 << ',' << patternspacing1 <<')'<<endl;
    cout << "Angle2 Init angle and spacing: (" << angle2 << ',' << patternspacing2 <<')'<<endl;
    cout << "Angle3 Init angle and spacing: (" << angle3 << ',' << patternspacing3 <<')'<<endl;



    cout << "Now handling img: "<< image_data_path << endl;
    cout << "Image Pages: " << images.size() << endl;



    // init recon class
    RECON_2D recon_2d(images, numangle, numphase, wl, na, pixel_size, 0.01, 0.001);




    /////////// reconstruction for all angle widefield 
    cout << "############  Reconstruct for All Angle Widefield ############# " << endl;
    vector<cv::Mat> allAngle_wf_recon = recon_2d.getallangleield(0);
    
    /////////// Save all angle widefield recon result

    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_TIFF_COMPRESSION);
    // compression_params.push_back(cv::TIFF_COMPRESSION_LZW);

    bool result = cv::imwrite("../data/allangle_wf_recon.tif", allAngle_wf_recon);
    // bool result = cv::imwrite("../data/allangle_wf_recon.tif", allAngle_wf_recon, compression_params);

    if (result)
    {std::cout << "Successfully saved image as tiff." << std::endl;}
    else
    {std::cout << "Failed to save image as tiff." << std::endl;}

    
    

    clock_t end = clock();
    double duration = (end - start) / (double)CLOCKS_PER_SEC;
    std::cout << "Time taken: " << duration << " seconds" << std::endl;





    return 0;

}