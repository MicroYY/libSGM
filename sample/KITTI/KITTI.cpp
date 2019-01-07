#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <time.h>

#include <opencv2/opencv.hpp>
#include <libsgm.h>

int main(int argc, char** argv)
{
	cv::Mat left, right;

	int disp_size = 64;
	if (argc >= 4) {
		disp_size = atoi(argv[3]);
	}
	int bits = 8;

	std::stringstream l, r;
	clock_t start, end;
	start = clock();
	for (size_t i = 0; i < 200; i++)
	{
		l.str("");
		r.str("");
		l << argv[1] << "/" << "image_2" << "/" << std::setw(6) << std::setfill('0') << 0 << "_10.png";
		r << argv[1] << "/" << "image_3" << "/" << std::setw(6) << std::setfill('0') << 0 << "_10.png";

		cv::Mat left = cv::imread(l.str(), 2);
		cv::Mat right = cv::imread(r.str(), 2);

		sgm::StereoSGM ssgm(left.cols, left.rows, disp_size, bits, 8, sgm::EXECUTE_INOUT_HOST2HOST);
		cv::Mat output(cv::Size(left.cols, left.rows), CV_8UC1);


		ssgm.execute(left.data, right.data, output.data);
		cudaDeviceSynchronize();

		l.str("");
		r.str("");
		l << argv[1] << "/" << "image_2" << "/" << std::setw(6) << std::setfill('0') << i << "_11.png";
		r << argv[1] << "/" << "image_3" << "/" << std::setw(6) << std::setfill('0') << i << "_11.png";


		
		left = cv::imread(l.str(), 2);
		right = cv::imread(r.str(), 2);

		//sgm::StereoSGM ssgm(left.cols, left.rows, disp_size, bits, 8, sgm::EXECUTE_INOUT_HOST2HOST);
		//cv::Mat output(cv::Size(left.cols, left.rows), CV_8UC1);
		ssgm.execute(left.data, right.data, output.data);
		cudaDeviceSynchronize();
	}
	end = clock();
	std::cout << (float(end - start) / (float)CLOCKS_PER_SEC)*1000.0 << "ms" << std::endl;
}