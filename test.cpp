#include <iostream>
#include <Eigen/Dense>
#include <chrono> 
#include <fstream>
#include <vector>
// #include <opencv2/opencv.hpp>

// #include "utils.hpp"
// #include "nms.hpp"

// using namespace cv;
using namespace std::chrono; 
using namespace Eigen;
using namespace std;


int main(){
    VectorXi a(5);
    a << 6,7,8,9,10;
    // const int rows = a.rows();
    cout << "a:\n" << a << endl;
    a.conservativeResize(3);
    cout << "a:\n" << a << endl;

    return 0;
}