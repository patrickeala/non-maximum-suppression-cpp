#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <chrono> 

using namespace std::chrono; 
using namespace std;
using namespace cv;
using namespace Eigen;

vector<vector<float>> y_pred(2006,vector<float>(33));
// Matrix<float, 2006, 33> y_pred_eigen;

  // Extracting bounding y_pred from .txt file
int main(){
  ifstream myReadFile;
  myReadFile.open("tflite_y_pred_raw.txt");
  auto start = high_resolution_clock::now(); 
  while (!myReadFile.eof()){
    for(int i = 0; i < 2006; i++){
      for (int j = 0; j < 33; j++){
        myReadFile >> y_pred[i][j];
      }
    }
  }
  auto stop = high_resolution_clock::now();
  // myReadFile.close("tflite_y_pred_raw.txt");
  auto duration = duration_cast<microseconds>(stop - start); 
  
  // To get the value of duration use the count() 
  // member function on the duration object 
  cout << "Hardcode duration: " << duration.count() << endl; 




  // // Print array
  // for(int i = 0; i<2006; i++) {
  //   for(int j = 0; j<33; j++) {
  //     cout << '(' << y_pred[i][j] << ")";
  //   }
  //   cout << "\n";
  // }

  // y_pred_decoded_raw = 




  // MatrixXd mat(3,4);
  // mat << 1,2,3,4,
  //        5,6,7,8,
  //        9,10,11,12;
  // MatrixXd ext;
  // ext = (mat.col(0).array() > 4).select(mat.row(), 0);
  // cout << ext << endl;




  // while (!myReadFile.eof()){
  //   for(int i = 0; i < 1; i++){
  //     vector<float> tmpVec;
  //     float tmpFloat;

  //     for (int j = 0; j < 5; j++){
  //       myReadFile  >> tmpFloat;
  //       tmpVec.push_back(tmpFloat);
  //     }
  //     y_pred.push_back(tmpVec);
  //     cout << "y_pred[" << i << "][" << j << "]: " << y_pred[i][j]; 

  //   }
  // }
  // cout << "shape of data: (" << y_pred[0].size() << "," << sizeof(y_pred).size() << ")\n";
  // cout << "y_pred[0][0]: " << y_pred[0][0] << "\n";
  return 0;
}













