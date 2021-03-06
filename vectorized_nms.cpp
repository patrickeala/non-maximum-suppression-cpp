#include "vectorized_nms.hpp"


vector<Rect> vectorized_nms(const MatrixXf & pred, const float & iou_threshold){
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

		// const float iou_threshold=0.45;
		VectorXf bool_keep = overlap;
		bool_keep = (overlap.array() <= iou_threshold).select(bool_keep, 1);


		
		// cout << "bool_keep: " << bool_keep << endl;
		// cout << "bool_keep.size(): " << bool_keep.size() << endl;


		int num_keep = (bool_keep.array()<1).count();

		// cout << "counting bool" <<(bool_keep.array()>0).count() << endl;

		VectorXi tmp_idxs = idxs;
		idxs.conservativeResize(num_keep, NoChange);
		// cout << "idxs.size(): " << idxs.size() << endl;



		int m=0;
		for (int n=0; n<bool_keep.size(); ++n){
			if(bool_keep(n)<1){
				idxs.row(m++) = tmp_idxs.row(n);
			}
		}
		// cout << "idxs:\n" << idxs << endl;
		// cout << "idxs.size(): " << idxs.size() << endl;


	}

	// cout << "pick:\n" << pick << endl;
	// cout << "pred shape: (" << pred.rows() << "," << pred.cols() << ")\n";
	
	int n_finalboxes = pick.rows();

	MatrixXf filtered(n_finalboxes,5);
	for (int x=0; x<n_finalboxes; x++){
		filtered.row(x) = pred.row(pick.data()[x]);
	}
	// cout << "filtered: " << filtered << endl;
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


    auto boxes_rect = VecBoxesToRectangles(final_boxes);


    return boxes_rect;
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
	// cout << "pick rows: " << row << endl;
	vect.conservativeResize(row + 1, NoChange);
    vect.row(row) << value;
}

VectorXf extract_values(VectorXf & vec, VectorXi & idxs){
	int n_idxs = idxs.size();
	VectorXf resultVec(n_idxs);
	for (int i=0; i<n_idxs; i++){
    	// resultVec.conservativeResize(resultVec.rows()+1,NoChange);
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
vector<Rect> VecBoxesToRectangles(const vector<vector<float>> & boxes)
{
  vector<Rect> rectangles;
  vector<float> box;
  
  for (const auto & box: boxes)
    rectangles.push_back(Rect(Point(box[1], box[2]), Point(box[3], box[4])));
  
  return rectangles;
}