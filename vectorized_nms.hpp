#ifndef VECTORIZED_NMS_HPP__
#define VECTORIZED_NMS_HPP__

#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <chrono> 
#include <fstream>

using std::vector;
using cv::Rect;
using cv::Point;
using namespace std;
using namespace Eigen;
using namespace std::chrono;


vector<Rect> vectorized_nms(const MatrixXf & boxes, const float & iou_thresh);

VectorXi argsort_eigen(VectorXf & vec);


void append_int_eigen(VectorXi & vect, int & value);

VectorXf extract_values(VectorXf & vec, VectorXi & idxs);

VectorXf max_eigen(VectorXf & vec1, int & i, VectorXf & vec2);

VectorXf min_eigen(VectorXf & vec1, int & i, VectorXf & vec2);

vector<Rect> VecBoxesToRectangles(const vector<vector<float>> & boxes);

#endif // VECTORIZED_NMS_HPP__