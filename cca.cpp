#include "cca.h"

CCA::CCA(cv::Mat &_X, cv::Mat &_Y){
	assert(_X.cols == _Y.cols);
	X = _X.clone();
	Y = _Y.clone();
	num_of_data = X.cols;

	center_x = calc_center(X);
	center_y = calc_center(Y);

	//std::cout << "center_x: "<< center_x << std::endl;
	//std::cout << "center_y: "<< center_y << std::endl;	

	centration(X);
	centration(Y);
}

CCA::~CCA(){

}

void CCA::calc(){

	//std::cout << X.rows << " " << X.cols << std::endl;	
	//std::cout << Y.rows << " " << Y.cols << std::endl;	

	//std::cout << X << std::endl;
	//std::cout << Y << std::endl;	

	Sxx = X*X.t();	
	Sxy = X*Y.t();	
	Syx = Y*X.t();
	Syy = Y*Y.t();

	//std::cout << "construct matrix S" << std::endl;
	cv::Mat Sa = Sxx.inv()*Sxy*Syy.inv()*Syx;
	cv::Mat Sb = Syy.inv()*Syx*Sxx.inv()*Sxy;	

	//std::cout << "SVD" << std::endl;
	cv::SVD svda(Sa);
	cv::SVD svdb(Sb);
	//std::cout << svda.u << std::endl;

	A = cv::Mat(X.rows, Y.rows, svda.u.type());
	for(int i=0; i<Y.rows; i++){
		A.col(i) = svda.u.col(i);
		svda.u.col(i).copyTo(A.col(i));
	}
	//std::cout << A.rows << " " << A.cols << std::endl;	
	B = svdb.u;		
	//std::cout << B.rows << " " << B.cols << std::endl;	
	cv::Mat diag = svda.w;
	S = cv::Mat::zeros(Y.rows, Y.rows, A.type());//svda.w;
	for(int i=0; i<Y.rows; i++){
		S.at<float>(i, i) = diag.at<float>(i, 0);
	}

	std::cout << A.t()*Sxx*A << std::endl;
	std::cout << B.t()*Syy*B << std::endl;	

	normalizeVariance(A, Sxx);
	normalizeVariance(B, Syy);	

	//std::cout << A << std::endl;
	//std::cout << B << std::endl;
	//std::cout << S << std::endl;

	G = B.t().inv()*S*A.t();
	//std::cout << G << std::endl;
/*
	cv::Mat Sxx_sqrt, Syy_sqrt;
	cv::sqrt(Sxx, Sxx_sqrt);
	cv::sqrt(Syy, Syy_sqrt);	
	cv::Mat S2 = Sxx_sqrt.inv()*Sxy*Syy_sqrt.inv();

	cv::SVD svd2(S2);
	A2 = svd2.u;
	B2 = svd2.vt;
	Diag2 = svd2.w;		
	*/
}

cv::Mat CCA::predict(const cv::Mat &x){
	//return A.t()*x*B.inv();
	/*
	std::cout << B.rows << " " << B.cols << std::endl;	
	std::cout << S.rows << " " << S.cols << std::endl;	
	std::cout << A.rows << " " << A.cols << std::endl;	
	std::cout << x.rows << " " << x.cols << std::endl;	
	*/

	//cv::Mat S_sqrt;
	//cv::sqrt(S, S_sqrt);

	//std::cout << S_sqrt << std::endl;

	//cv::Mat G = B.t().inv()*S*A.t();
	//cv::Mat G = B.t().inv()*S_sqrt*A.t();

	//std::cout << "G: " << G << std::endl;

	return G*(x-center_x)+center_y;
}

cv::Mat CCA::calc_center(const cv::Mat &X){
	cv::Mat one = cv::Mat::ones(X.cols, 1, CV_32FC1);
	cv::Mat center = X*one;
	center /= X.cols;
	return center;
}

void CCA::centration(cv::Mat &X){
	cv::Mat center = calc_center(X);
	for(int i=0; i<X.cols; i++){
		X.col(i) -= center;
	}
}

void CCA::normalizeVariance(cv::Mat &X, const cv::Mat Sigma){
	cv::Mat V = X.t()*Sigma*X;
	for(int i=0; i<X.cols; i++){
		std::cout << V.at<float>(i, i) << std::endl;
		X.col(i) /= V.at<float>(i, i);
	}
}