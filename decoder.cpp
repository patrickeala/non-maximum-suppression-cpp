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

MatrixXf convert_coordinates(const MatrixXf & matrix){
	// cout << "Starting cc\n";
	MatrixXf converted = matrix;
	int start_index = matrix.cols();
	// cout << "Received converted\n";
	converted.col(start_index-4) = matrix.col(start_index-4) - matrix.col(start_index-2) / 2;
	converted.col(start_index-3) = matrix.col(start_index-3) - matrix.col(start_index-1) / 2;
	converted.col(start_index-2) = matrix.col(start_index-4) + matrix.col(start_index-2) / 2;
	converted.col(start_index-1) = matrix.col(start_index-3) + matrix.col(start_index-1) / 2;
	// cout << "Finishing cc\n";
	return converted;
}

int main()
{
	//FIX THIS
	// const int y_pred_rows = y_pred.rows();
	// const int y_pred_cols = y_pred.cols();
	// const int y_rows = y_pred_rows;
	// const int y_cols = y_pred_cols;

	const int y_pred_rows = 2006;
	const int y_pred_cols = 33;
	const int y_raw_cols = y_pred_cols - 8;
	const int img_height = 300;
	const int img_width = 300;
	const int n_classes = y_raw_cols-4;
	const float confidence_thresh = 0.5;


  MatrixXf y_pred(y_pred_rows, y_pred_cols);
  ifstream myReadFile;
  myReadFile.open("tflite_y_pred_raw.txt");

  MatrixXf y_pred_decoded_raw(y_pred_rows,y_pred_cols-8);

  //Extracting into eigen matrix
  auto start = high_resolution_clock::now(); 
  while (!myReadFile.eof()){
    for(int i = 0; i < y_pred_rows; i++){
      for (int j = 0; j < y_pred_cols; j++){
        // myReadFile >> y_pred[i][j];
        myReadFile >> y_pred(i,j);

      }
    }
  }
  auto stop = high_resolution_clock::now();

  auto duration = duration_cast<microseconds>(stop - start); 
  
  // To get the value of duration use the count() 
  // member function on the duration object 
  // cout << y_pred << endl;

  // cout << "Eigen duration: " << duration.count() << endl;
	// cout << "y_pred shape: (" << y_pred.rows() << "," << y_pred.cols() << ")\n";
	// cout << "y_pred rows,cols: (" << y_pred_rows << "," << y_pred_cols << ")\n";


	y_pred_decoded_raw = y_pred.block<y_pred_rows, y_raw_cols>(0,0);
	// y_pred_decoded_raw = y_pred(all, seq(0,25));
	
	// y_pred_decoded = A.col(A.cols()-2)
	// cout << "y_pred_decoded_raw shape: (" << y_pred_decoded_raw.rows() << "," << y_pred_decoded_raw.cols() << ")\n";


	// 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates
	y_pred_decoded_raw.block<y_pred_rows,2>(0,y_raw_cols-2) = (y_pred_decoded_raw.block<y_pred_rows,2>(0,y_raw_cols-2).array() * y_pred.block<y_pred_rows,2>(0,y_pred_cols-2).array()).exp();
	y_pred_decoded_raw.block<y_pred_rows,2>(0,y_raw_cols-2) = y_pred_decoded_raw.block<y_pred_rows,2>(0,y_raw_cols-2).array() * y_pred.block<y_pred_rows,2>(0,y_pred_cols-6).array();
	y_pred_decoded_raw.block<y_pred_rows,2>(0,y_raw_cols-4) = y_pred_decoded_raw.block<y_pred_rows,2>(0,y_raw_cols-4).array() * y_pred.block<y_pred_rows,2>(0,y_pred_cols-4).array() * y_pred.block<y_pred_rows,2>(0,y_pred_cols-6).array();
	y_pred_decoded_raw.block<y_pred_rows,2>(0,y_raw_cols-4) = y_pred_decoded_raw.block<y_pred_rows,2>(0,y_raw_cols-4).array() + y_pred.block<y_pred_rows,2>(0,y_pred_cols-8).array();
  // cout << y_pred_decoded_raw.block<y_pred_rows,2>(0,y_raw_cols-4) << endl;

	y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw);

	// Normalizing coordinates
	y_pred_decoded_raw.col(y_raw_cols-1) *= img_height;
	y_pred_decoded_raw.col(y_raw_cols-3) *= img_height;
	y_pred_decoded_raw.col(y_raw_cols-2) *= img_width;
	y_pred_decoded_raw.col(y_raw_cols-4) *= img_width;



  // cout << y_pred_decoded_raw << endl;
	cout << "n_classes: " << n_classes << endl;
	Matrix <float, Dynamic, 5> pred;
	for (int class_id = 1; class_id < n_classes; class_id++){

			
		MatrixXf single_class(y_pred_rows,5);
		single_class.col(0) = y_pred_decoded_raw.col(class_id);
		single_class.block<y_pred_rows,4>(0,1) = y_pred_decoded_raw.block<y_pred_rows,4>(0,y_raw_cols-4);
	  	
		VectorXf Thresh_Met(y_pred_rows,1);
		Thresh_Met = single_class.col(0);
		Thresh_Met = (Thresh_Met.array() > confidence_thresh).select(Thresh_Met, 0);
		Matrix <bool, y_pred_rows, 1> non_zeros = Thresh_Met.cast<bool>().rowwise().any();
		MatrixXf threshold_met(non_zeros.count(),5);

		int j=0;
		for (int i=0 ; i<y_pred_rows ; ++i){
			if (non_zeros(i)){
				threshold_met.row(j++) = single_class.row(i);
			}
		}
		// cout << threshold_met << endl;

		int thresh_rows = threshold_met.rows();
		if(thresh_rows!=0){
			int pred_rows = pred.rows(); 		

			cout << "trying to append " << class_id << endl;
			MatrixXf tmp(pred_rows + thresh_rows,5);
			tmp << pred, threshold_met;
			// pred.conservativeResize(pred_rows + thresh_rows, NoChange);
			// pred.block<thresh_rows,5>(pred_rows,0) = threshold_met;
			pred = tmp;
		}
	}
	cout << pred << endl;
	cout << "pred shape: (" << pred.rows() << "," << pred.cols() << ")\n";

	// convert threshold met to 2D vec
	// vector<vector<float>> boxes(pred.rows(),vector<float>(5));
	vector<vector<float>> boxes(pred.rows(),vector<float>(5));

	// for (int i=0; i<pred.rows(); ++i){
    	// const float* begin = &pred.row(i).data()[0];
    	// boxes.push_back(vector<float>(begin, begin+pred.cols()));
	// }
	// Eigen::Map<MatrixXf>(&boxes[0][0], 52, 5) = pred;
	for(int i = 0; i<52; i++) {
    	for(int j = 0; j<5; j++) {
      		boxes[i][j] = pred[i][j];
    	}
  	}

	for(int i = 0; i<52; i++) {
    	for(int j = 0; j<5; j++) {
      	cout << " " << boxes[i][j] << " ";
    	}
    	cout << "\n";
  	}

	// // initialization
	// Mat imgBefore(Size(500, 500), DataType<float>::type);
  	// // Mat imgBefore("silicon_valley.jpg");

  	// Mat imgAfter = imgBefore.clone();
  	// float threshold	= 0.5;
	// DrawRectangles(imgBefore, boxes);
	// imshow("Before", imgBefore);
	
	// // after
	// vector<Rect> reducedRectangle = nms(boxes, threshold);
	// DrawRectangles(imgAfter, reducedRectangle);
	// imshow("After", imgAfter);
	
	// waitKey(0);

}




