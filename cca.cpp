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
	//std::cout << Sxx << std::endl;
	std::cout << "Sxx.rows: " << Sxx.rows << " cols: " << Sxx.cols << std::endl;
	Sxy = X*Y.t();	
	//std::cout << Sxy << std::endl;	
	std::cout << "Sxy.rows: " << Sxy.rows << " cols: " << Sxy.cols << std::endl;	
	Syx = Y*X.t();
	//std::cout << Syx << std::endl;	
	std::cout << "Syx.rows: " << Syx.rows << " cols: " << Syx.cols << std::endl;	
	Syy = Y*Y.t();
	//std::cout << Syy << std::endl;		
	std::cout << "Syy.rows: " << Syy.rows << " cols: " << Syy.cols << std::endl;	

	std::cout << "construct matrix S" << std::endl;
	cv::Mat Sb = Syy.inv()*Syx*Sxx.inv()*Sxy;
	cv::Mat Sa = Sxx.inv()*Sxy*Syy.inv()*Syx;

	std::cout << "SVD" << std::endl;
	cv::SVD svda(Sa);
	cv::SVD svdb(Sb);

	A = svda.u;
	std::cout << "A.rows: " << A.rows << " cols: " << A.cols << std::endl;
	B = svdb.u;		
	std::cout << "B.rows: " << B.rows << " cols: " << B.cols << std::endl;	

	std::cout << A << std::endl;
	std::cout << B << std::endl;	

	//std::cout << "A.rows: " << A.rows << " cols: " << A.cols << std::endl;	
	//std::cout << "B.rows: " << B.rows << " cols: " << B.cols << std::endl;		
}

cv::Mat CCA::predict(cv::Mat x){
	//std::cout << x.rows << " " << x.cols << std::endl;
	return A.t()*x/B.at<float>(0, 0);
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