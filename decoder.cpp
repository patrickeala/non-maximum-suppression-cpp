#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

vector<vector<float>> y_pred(2006,vector<float>(33));


  // Extracting bounding y_pred from .txt file
int main(){
  ifstream myReadFile;
  myReadFile.open("tflite_y_pred_raw.txt");

  while (!myReadFile.eof()){
    for(int i = 0; i < 2006; i++){
      for (int j = 0; j < 33; j++){
        myReadFile >> y_pred[i][j];
      }
    }
  }

  // Print array
  for(int i = 0; i<2006; i++) {
    for(int j = 0; j<33; j++) {
      cout << '(' << y_pred[i][j] << ")";
    }
    cout << "\n";
  }

  y_pred_decoded_raw = 










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













