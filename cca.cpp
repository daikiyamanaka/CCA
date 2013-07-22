#include "cca.h"

CCA::CCA(cv::Mat &_X, cv::Mat &_Y){
	assert(_X.cols == _Y.cols);
	X = _X.clone();
	Y = _Y.clone();
	num_of_data = X.cols;

//	centration(X);
//	centration(Y);	
}

CCA::~CCA(){

}

void CCA::calc(){

	Sxx = X*X.t();	
	Sxy = X*Y.t();	
	Syx = Y*X.t();
	Syy = Y*Y.t();

	//std::cout << "construct matrix S" << std::endl;
	cv::Mat Sb = Syy.inv()*Syx*Sxx.inv()*Sxy;
	cv::Mat Sa = Sxx.inv()*Sxy*Syy.inv()*Syx;

	//std::cout << "SVD" << std::endl;
	cv::SVD svda(Sa);
	cv::SVD svdb(Sb);

	A = svda.u;	
	std::cout << A.rows << " " << A.cols << std::endl;	
	B = svdb.u;		
	std::cout << B.rows << " " << B.cols << std::endl;	
	cv::Mat diag = svda.w;
	S = cv::Mat::zeros(A.rows, A.cols, A.type());//svda.w;
	for(int i=0; i<diag.rows; i++){
		S.at<float>(i, i) = diag.at<float>(i, 0);
	}
	std::cout << S.rows << " " << S.cols << std::endl;
}

cv::Mat CCA::predict(cv::Mat x){
	//return A.t()*x*B.inv();
	return B.inv()*S*A.t()*x;
}

cv::Mat CCA::calc_center(cv::Mat &X){
	cv::Mat one = cv::Mat::ones(X.cols, 1, CV_32FC1);
	cv::Mat center = X*one;
	center /= X.cols;
	return center;
}

void CCA::centration(cv::Mat &X){
	cv::Mat center = calc_center(X);
}