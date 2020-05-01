#include <iostream>
#include <fstream>

#include <vector>
#include <opencv2/opencv.hpp>

#include "utils.hpp"
#include "nms.hpp"

using namespace cv;
using namespace std;

int main()
{
  // initialization
  Mat imgBefore(Size(500, 500), DataType<float>::type);
  // Mat imgBefore("silicon_valley.jpg");

  Mat imgAfter = imgBefore.clone();
  float threshold	= 0.5;


  // Extracting bounding boxes from .txt file

  ifstream myReadFile;
  myReadFile.open("tflite_y_pred_before_nms.txt");
  vector<vector<float>> boxes(52,vector<float>(5));
  
  while (!myReadFile.eof()){
    for(int i = 0; i < 52; i++){
      for (int j = 0; j < 5; j++){
        myReadFile >> boxes[i][j];
      }
    }
  }
  for(int i = 0; i<52; i++) {
    for(int j = 0; j<5; j++) {
      cout << '(' << boxes[i][j] << ")";
    }
    cout << "\n";
  }

  // vector<vector<float>> boxes;
  // while (!myReadFile.eof()){
  //   for(int i = 0; i < 1; i++){
  //     vector<float> tmpVec;
  //     float tmpFloat;
  //     for (int j = 0; j < 5; j++){
  //       myReadFile  >> tmpFloat;
  //       tmpVec.push_back(tmpFloat);
  //     }
  //     boxes.push_back(tmpVec);
  //   }
  // }
  cout << "shape of boxes: (" << boxes[0].size() << "," << sizeof(boxes).size() << ")\n";
  // vector<vector<float> > rectangles =
  // {
  //   {300, 300, 400, 400},
  //   {320, 320, 420, 420},
  //   {295, 259, 415, 415},
  //   {100, 100, 150, 150},
  //   {90,  90,  180, 180},
  //   {112, 112, 170, 170}
  // };

  // before
  DrawRectangles(imgBefore, boxes);
  imshow("Before", imgBefore);
  
  // after
  vector<Rect> reducedRectangle = nms(boxes, threshold);
  DrawRectangles(imgAfter, reducedRectangle);
  imshow("After", imgAfter);
  
  waitKey(0);
}