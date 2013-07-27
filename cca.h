#include <opencv2/opencv.hpp>

class CCA
{
public:
	CCA(cv::Mat &X, cv::Mat &Y);
	~CCA();

	void calc();
	cv::Mat predict(const cv::Mat &x);

private:
	cv::Mat calc_center(const cv::Mat &X); // calc col_wise center of mass
	void centration(cv::Mat &X);
	void normalizeVariance(cv::Mat &X, const cv::Mat Sigma);

	cv::Mat X, Y;
	cv::Mat A, B, S, G;
	cv::Mat Sxx, Sxy, Syx, Syy;
	cv::Mat center_x, center_y;
	int num_of_data;

	//cv::Mat A2, B2, Diag2;
};