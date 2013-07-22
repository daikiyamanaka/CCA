#include <opencv2/opencv.hpp>

class CCA
{
public:
	CCA(cv::Mat &X, cv::Mat &Y);
	~CCA();

	void calc();
	cv::Mat predict(cv::Mat x);

private:
	cv::Mat calc_center(cv::Mat &X); // calc col_wise center of mass
	void centration(cv::Mat &X);
	cv::Mat X, Y;
	cv::Mat A, B;
	cv::Mat Sxx, Sxy, Syx, Syy;
	int num_of_data;
	/* data */
};