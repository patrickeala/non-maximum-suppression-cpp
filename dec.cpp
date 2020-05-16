#include "dec.hpp"

MatrixXf convert_coordinates(const MatrixXf & matrix){
	MatrixXf converted = matrix;
	int start_index = matrix.cols();
	converted.col(start_index-4) = matrix.col(start_index-4) - matrix.col(start_index-2) / 2;
	converted.col(start_index-3) = matrix.col(start_index-3) - matrix.col(start_index-1) / 2;
	converted.col(start_index-2) = matrix.col(start_index-4) + matrix.col(start_index-2) / 2;
	converted.col(start_index-1) = matrix.col(start_index-3) + matrix.col(start_index-1) / 2;
	cout << "Finishing cc\n";
	return converted;
}

MatrixXf decode_detections(const MatrixXf & y_pred, const float & confidence_thresh, const float & iou_threshold, const int & top_k, const int & img_height, const int & img_width){
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


	return pred;
}