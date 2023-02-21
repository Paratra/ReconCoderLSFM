// COPYRIGHT: Ming Song UGA Kner Lab 2023
// __function__: SIM 2D reconstruction
// __author__:  MingSong
#include "Utility.h"



template<typename T>
std::vector<T> linespace(T start, T stop, T num) {
    if (stop < start)
    {
       cout<< "Stop is smaller than Start.";
       abort();
    }
    
    double step = round((stop-start)/num);

    std::vector<T> values;
    for (T value = start; value <= stop; value += step){
        
        values.push_back(value);
    }
    return values;
}


MeshGrid_struct meshgrid(vector<double> t_x, vector<double> t_y) {
    // std::vector<int> t_x, t_y;
    // for (int i = xgv.start; i <= xgv.end; i += xgv.step) t_x.push_back(i);
    // for (int i = ygv.start; i <= ygv.end; i += ygv.step) t_y.push_back(i);

    cv::Mat XX = cv::Mat(t_y.size(), t_x.size(), CV_32F);
    cv::Mat YY = cv::Mat(t_y.size(), t_x.size(), CV_32F);

    for (int i = 0; i < t_y.size(); i++) {
        for (int j = 0; j < t_x.size(); j++) {
        XX.at<float>(i, j) = t_x[j];
        YY.at<float>(i, j) = t_y[i];
        }
    }
    MeshGrid_struct meshgrid_result;
    meshgrid_result.X = XX;
    meshgrid_result.Y = YY;

    return meshgrid_result;
}




cv::Mat discArray(vector<double> shape, double radius, vector<double> origin){

    double nx = shape[0];
    double ny = shape[1];

    double ox = nx/2;
    double oy = ny/2;

    vector<double> x = linespace(-ox, ox-1, nx);
    vector<double> y = linespace(-oy, oy-1, ny);

    MeshGrid_struct Mesh_XY = meshgrid(y, x);

    cv::Mat X = Mesh_XY.X;
    cv::Mat Y = Mesh_XY.Y;

    cv::Mat rho(X.size(), X.type());
    for (int i = 0; i < X.rows; i++) {
        for (int j = 0; j < X.cols; j++) {
        float x = X.at<float>(i, j);
        float y = Y.at<float>(i, j);
        rho.at<float>(i, j) = std::sqrt(x * x + y * y);
        }
    }

    cv::Mat disc(rho.size(), rho.type());
    for (int i = 0; i < rho.rows; i++) {
        for (int j = 0; j < rho.cols; j++) {
            if (rho.at<float>(i, j) >= radius){
                disc.at<float>(i, j) = 0;
            }else{
                disc.at<float>(i, j) = 1;
            }
        }
    }

    
    if (origin[0] != -1 && origin[1] != -1){
        int s0 = origin[0] - nx/2;
        int s1 = origin[1] - ny/2;
        // [TODO]: np.roll in c++
    }
    


    // cout << disc.size() << endl;
    // cout << disc(cv::Range(0,5),cv::Range(0,5)) << endl;

    return disc;

}



// int main() {

//     return 0;
// }