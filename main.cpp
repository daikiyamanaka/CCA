#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "cca.h"

std::vector<Eigen::Vector4f > iris_data;
std::vector<float> iris_type;

std::vector<std::string> split(const std::string &str, char delim){
  std::istringstream iss(str); std::string tmp; std::vector<std::string> res;
  while(getline(iss, tmp, delim)) res.push_back(tmp);
  return res;
}

void importIrisData(std::string filename){
	std::ifstream ifs;
	std::string line;

	ifs.open(filename.c_str());
	if(!ifs){
		std::cerr << "Can't read input file " << std::endl;
	}

	//Eigen::Vector4f average = Eigen::Vector4f::Zero();

	std::cout << "reading" << std::endl;
	while(getline(ifs, line)){
		Eigen::Vector4f iris;
		std::vector<std::string> strs = split(line, ',');

		if(strs.size() == 0){
			continue;
		}

		for(int i=0; i<4; i++){
			iris[i] = atof(strs[i].c_str());
		}
		iris_data.push_back(iris);
		//average += iris;
		if(strs[4] == "Iris-setosa"){
			iris_type.push_back(1.0);
		}
		else if(strs[4] == "Iris-versicolor"){
			iris_type.push_back(2.0);
		}
		else if(strs[4] == "Iris-virginica"){
			iris_type.push_back(3.0);
		}
		else{
			iris_type.push_back(-1.0);
		}
	}
	/*
	average = average/(float)iris_data.size();
	for(int i=0; i<iris_data.size(); i++){
		iris_data[i] -= average;
	}
	*/
	ifs.close();
}
cv::Mat calc_center(cv::Mat &X){
	cv::Mat one = cv::Mat::ones(X.cols, 1, CV_32FC1);
	cv::Mat center = X*one;
	center /= X.cols;
	return center;
}

void centration(cv::Mat &X){
	cv::Mat center = calc_center(X);
	//std::cout << center << std::endl;
	for(int i=0; i<X.cols; i++){
		X.col(i) -= center;
	}
}


int main(int argc, char const *argv[])
{
	importIrisData("../iris.data");

	cv::Mat iris_data_mat = cv::Mat(4, iris_data.size(), CV_32FC1);
	cv::Mat iris_label_mat = cv::Mat(1, iris_type.size(), CV_32FC1);	

	for(int i=0; i<iris_data.size(); i++){
		iris_data_mat.at<float>(0, i) =	iris_data[i][0];
		iris_data_mat.at<float>(1, i) =	iris_data[i][1];
		iris_data_mat.at<float>(2, i) =	iris_data[i][2];
		iris_data_mat.at<float>(3, i) =	iris_data[i][3];						
		iris_label_mat.at<float>(0, i) = iris_type[i];
	}

	centration(iris_data_mat);
	centration(iris_label_mat);

	std::cout << iris_data_mat << std::endl;
	std::cout << iris_label_mat << std::endl;	

	CCA cca(iris_data_mat, iris_label_mat);
	std::cout << "calc" << std::endl;
	cca.calc();

	for(int i=0; i<iris_data.size(); i++){
		cv::Mat pre = cca.predict(iris_data_mat.col(i));
		std::cout << "true: " << iris_type[i] << " predict: " << pre.at<float>(0, 0) << std::endl;
	}

	return 0;
}