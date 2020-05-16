#include <iostream>
#include <Eigen/Dense>
#include <chrono> 
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "utils.hpp"
#include "nms.hpp"

using namespace cv;
using namespace std::chrono; 
using namespace Eigen;
using namespace std;


int main(){
    VectorXi vec = VectorXi::LinSpaced(5, 0, 4);
    VectorXi a = VectorXi::LinSpaced(5, 6, 10);
    a << 6,7,8,9,10;
    cout << vec << endl;
    cout << a << endl;
    // vec.conservativeResize(vec.rows()-1,NoChange);
    // vec.row(4) << 1;
    // cout << vec << endl;
    // a = vec;
    
    cout << "vec*a: " << vec.array()*a.array()<< endl;

    return 0;
}