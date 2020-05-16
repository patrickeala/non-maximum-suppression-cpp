#include <iostream>
#include <Eigen/Dense>
#include <chrono> 
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <typeinfo>

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

vector<vector<float>> decode_detections(const MatrixXf & y_pred, const float & confidence_thresh=0.3, const float & iou_threshold=0.45, const int & top_k=200, const int & img_height=300, const int & img_width=300){
	const int y_pred_rows = 2006;
	const int y_pred_cols = 33;
	const int y_raw_cols = y_pred_cols - 8;
	// const int img_height = 300;
	// const int img_width = 300;
	const int n_classes = y_raw_cols-4;
	// const float confidence_thresh = 0.5;
	
	MatrixXf y_pred_decoded_raw(y_pred_rows,y_pred_cols-8);
	Matrix <float, Dynamic, 5> pred;

	

	y_pred_decoded_raw = y_pred.block<y_pred_rows, y_raw_cols>(0,0);
	y_pred_decoded_raw.block<y_pred_rows,2>(0,y_raw_cols-2) = (y_pred_decoded_raw.block<y_pred_rows,2>(0,y_raw_cols-2).array() * y_pred.block<y_pred_rows,2>(0,y_pred_cols-2).array()).exp();
	y_pred_decoded_raw.block<y_pred_rows,2>(0,y_raw_cols-2) = y_pred_decoded_raw.block<y_pred_rows,2>(0,y_raw_cols-2).array() * y_pred.block<y_pred_rows,2>(0,y_pred_cols-6).array();
	y_pred_decoded_raw.block<y_pred_rows,2>(0,y_raw_cols-4) = y_pred_decoded_raw.block<y_pred_rows,2>(0,y_raw_cols-4).array() * y_pred.block<y_pred_rows,2>(0,y_pred_cols-4).array() * y_pred.block<y_pred_rows,2>(0,y_pred_cols-6).array();
	y_pred_decoded_raw.block<y_pred_rows,2>(0,y_raw_cols-4) = y_pred_decoded_raw.block<y_pred_rows,2>(0,y_raw_cols-4).array() + y_pred.block<y_pred_rows,2>(0,y_pred_cols-8).array();

	y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw);

	// Normalizing coordinates
	y_pred_decoded_raw.col(y_raw_cols-1) *= img_height;
	y_pred_decoded_raw.col(y_raw_cols-3) *= img_height;
	y_pred_decoded_raw.col(y_raw_cols-2) *= img_width;
	y_pred_decoded_raw.col(y_raw_cols-4) *= img_width;


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

			// cout << "trying to append " << class_id << endl;
			MatrixXf tmp(pred_rows + thresh_rows,5);
			tmp << pred, threshold_met;
			// pred.conservativeResize(pred_rows + thresh_rows, NoChange);
			// pred.block<thresh_rows,5>(pred_rows,0) = threshold_met;
			pred = tmp;
		}
	}

	vector<vector<float>> boxes(pred.rows(),vector<float>(5));
	float array[pred.rows()][5];

	// converted to array
	for (int i=0 ; i<pred.rows() ; i++){
		Map<RowVectorXf>(&array[i][0], 1, 5) = pred.row(i);
	}
	// converted to 2dvec
    for(int i = 0; i < pred.rows(); i++){
      for (int j = 0; j < 5; j++){
        boxes[i][j] = array[i][j];
      }
    }


	return boxes;
}

VectorXi argsort_eigen(VectorXf & vec){
	// Initialize indices
	VectorXi idxs = VectorXi::LinSpaced(vec.size(), 0, vec.size()-1);
	// Sort indices
	sort(idxs.data(), idxs.data()+idxs.size(),[&vec](int i1, int i2) {return vec.data()[i1] < vec.data()[i2];});

	return idxs;
}


void append_int_eigen(VectorXi & vect, int & value)
{
    int row = vect.rows();
	cout << "pick rows: " << row << endl;
	vect.conservativeResize(row + 1, NoChange);
    vect.row(row) << value;
}

VectorXf extract_values(VectorXf & vec, VectorXi & idxs){
	VectorXf resultVec;
	for (int i=0; i<idxs.size(); i++){
    	resultVec.conservativeResize(resultVec.rows()+1,NoChange);
		// int a = idxs.data()[i];
		resultVec.row(i) << vec.row(idxs.data()[i]);
	}
  return resultVec;
}

VectorXf max_eigen(VectorXf & vec1, int & i, VectorXf & vec2){
	VectorXf maxVec = vec2;
	float x = vec1.data()[i];
	maxVec = (maxVec.array() >= x).select(maxVec, x); 
	return maxVec;
}

VectorXf min_eigen(VectorXf & vec1, int & i, VectorXf & vec2){
	VectorXf maxVec = vec2;
	float x = vec1.data()[i];
	maxVec = (maxVec.array() <= x).select(maxVec, x); 
	return maxVec;
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
  	const float threshold = 0.5;



  MatrixXf y_pred(y_pred_rows, y_pred_cols);
  ifstream myReadFile;
  myReadFile.open("tflite_y_pred_raw.txt");

	auto start = high_resolution_clock::now();

  MatrixXf y_pred_decoded_raw(y_pred_rows,y_pred_cols-8);

  //Extracting into eigen matrix
  while (!myReadFile.eof()){
    for(int i = 0; i < y_pred_rows; i++){
      for (int j = 0; j < y_pred_cols; j++){
        // myReadFile >> y_pred[i][j];
        myReadFile >> y_pred(i,j);

      }
    }
  }
	// vector<vector<float>> boxes = decode_detections(y_pred);




//   auto start = high_resolution_clock::now(); 

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
	// cout << "n_classes: " << n_classes << endl;
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

			// cout << "trying to append " << class_id << endl;
			MatrixXf tmp(pred_rows + thresh_rows,5);
			tmp << pred, threshold_met;
			// pred.conservativeResize(pred_rows + thresh_rows, NoChange);
			// pred.block<thresh_rows,5>(pred_rows,0) = threshold_met;
			pred = tmp;
		}
	}

	// // convert threshold met to 2D vec
	// auto conv_start = high_resolution_clock::now(); 

	// vector<vector<float>> boxes(pred.rows(),vector<float>(5));
	// float array[pred.rows()][5];

	// // converted to array
	// for (int i=0 ; i<pred.rows() ; i++){
	// 	Map<RowVectorXf>(&array[i][0], 1, 5) = pred.row(i);
	// }
	// // converted to 2dvec
    // for(int i = 0; i < pred.rows(); i++){
    //   for (int j = 0; j < 5; j++){
    //     boxes[i][j] = array[i][j];
    //   }
    // }
	// auto conv_stop = high_resolution_clock::now();
	// auto conv_duration = duration_cast<nanoseconds>(conv_stop - conv_start); 
	// cout << "Conversion duration: " << conv_duration.count()/1e+6 << "ms" << endl;


	// cout << "pred:\n"  << pred << endl;
	// cout << "pred shape: ("  << pred.rows() << "," << pred.cols() << ")" << endl;

//
// START OF NMS
//

	VectorXf conf = pred.col(0);
    VectorXf x1 = pred.col(1);
    VectorXf y1 = pred.col(2);
    VectorXf x2 = pred.col(3);
    VectorXf y2 = pred.col(4);
    VectorXf area;



	// Compute Area
    area = ((x2 - x1).array() + 1) * ((y2 - y1).array() + 1);




	// FIX THIS: ARGSORT CONF TO GET IDXS

	VectorXi idxs = argsort_eigen(conf);
	// No need to sort confs
	// sort(conf.data(),conf.data()+conf.size());


	// cout << "conf:\n" << conf << endl;
	// cout << "idxs:\n" << idxs << endl;


	// BEGIN NMS LOOP
	int last;
	int i;
	VectorXi pick;


	while (idxs.size() > 0){
	
		last = idxs.size() - 1;
		i = idxs[last];
		
		// Remove last from idxs
		VectorXi idxs_no_last;
		idxs_no_last = idxs;
		idxs_no_last.conservativeResize(idxs_no_last.rows()-1,NoChange);
		
		
		append_int_eigen(pick, i);
		// cout << "pick: " << pick << endl;
		
		VectorXf extracted_x1 = extract_values(x1, idxs_no_last);
		VectorXf xx1 = max_eigen(x1, i, extracted_x1);

		VectorXf extracted_y1 = extract_values(y1, idxs_no_last);
		VectorXf yy1 = max_eigen(y1, i, extracted_y1);

		VectorXf extracted_x2 = extract_values(x2, idxs_no_last);
		VectorXf xx2 = min_eigen(x2, i, extracted_x2);

		VectorXf extracted_y2 = extract_values(y2, idxs_no_last);
		VectorXf yy2 = min_eigen(y2, i, extracted_y2);

		// cout << "xx1: " << xx1 << endl;
		

		// compute height and width
		VectorXf w = (xx2 - xx1).array() + 1;
		w = (w.array() >= 0).select(w, 0); 
		
		VectorXf h = (yy2 - yy1).array() + 1;
		h = (h.array() >= 0).select(h, 0); 
		

		VectorXf extracted_area = extract_values(area, idxs_no_last);
		VectorXf overlap = (w.array()*h.array()) / extracted_area.array();

		// cout << "overlap: " << overlap << endl;
		// cout << "overlap.size(): " << overlap.size() << endl;

		const float iou_threshold=0.45;
		VectorXf bool_keep = overlap;
		bool_keep = (overlap.array() <= iou_threshold).select(bool_keep, 1);


		
		cout << "bool_keep: " << bool_keep << endl;
		cout << "bool_keep.size(): " << bool_keep.size() << endl;


		int num_keep = (bool_keep.array()<1).count();

		// cout << "counting bool" <<(bool_keep.array()>0).count() << endl;

		VectorXi tmp_idxs = idxs;
		idxs.conservativeResize(num_keep, NoChange);
		cout << "idxs.size(): " << idxs.size() << endl;



		int m=0;
		for (int n=0; n<bool_keep.size(); ++n){
			if(bool_keep(n)<1){
				idxs.row(m++) = tmp_idxs.row(n);
			}
		}
		cout << "idxs:\n" << idxs << endl;
		cout << "idxs.size(): " << idxs.size() << endl;


	}

	cout << "pick:\n" << pick << endl;
	cout << "pred shape: (" << pred.rows() << "," << pred.cols() << ")\n";
	
	int n_finalboxes = pick.rows();

	MatrixXf filtered(n_finalboxes,5);
	for (int x=0; x<n_finalboxes; x++){
		filtered.row(x) = pred.row(pick.data()[x]);
	}
	cout << "filtered: " << filtered << endl;
//
//	Conversion to array
//


	vector<vector<float>> final_boxes(n_finalboxes,vector<float>(5));
	float array_final[n_finalboxes][5];

	// converted to array
	for (int i=0 ; i<n_finalboxes ; i++){
		Map<RowVectorXf>(&array_final[i][0], 1, 5) = filtered.row(i);
	}
	// converted to 2dvec
    for(int i = 0; i < n_finalboxes; i++){
      for (int j = 0; j < 5; j++){
        final_boxes[i][j] = array_final[i][j];
      }
    }

	for (int i=0; i<n_finalboxes; i++){
		for (int j=0; j<5; j++){
			cout << final_boxes[i][j] << " ";
		}
		cout << "\n";
	}








	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<nanoseconds>(stop - start); 
	cout << "Vectorized NMS duration: " << duration.count() << endl; 






//
// OLD NMS
//
	
	// vector<Rect> reducedRectangle = nms(boxes, threshold);

	// Mat imgBefore(Size(500, 500), DataType<float>::type);
  	// // Mat imgBefore("silicon_valley.jpg");

  	// Mat imgAfter = imgBefore.clone();	
	// DrawRectangles(imgBefore, boxes);
	// imshow("Before", imgBefore);
	// DrawRectangles(imgAfter, reducedRectangle);
	// imshow("After", imgAfter);
	
	// waitKey(0);

}






