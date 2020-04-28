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
  vector<vector<float>> boxes;

  while (!myReadFile.eof()){
    for(int i = 0; i < 1; i++){
      vector<float> tmpVec;
      float tmpFloat;

      for (int j = 0; j < 5; j++){
        myReadFile  >> tmpFloat;
        tmpVec.push_back(tmpFloat);
      }
      boxes.push_back(tmpVec);
    }
  }

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