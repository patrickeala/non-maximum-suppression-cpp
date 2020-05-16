#ifndef DEC_HPP__
#define DEC_HPP__ 

#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;


MatrixXf convert_coordinates(const MatrixXf & matrix);

MatrixXf decode_detections(const MatrixXf & y_pred, const float & confidence_thresh=0.3, const float & iou_threshold=0.45, const int & top_k=200, const int & img_height=300, const int & img_width=300);



#endif // DEC_HPP__ 