// LaneDetection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "LaneDetector.h"


void testImage(const std::string &fname)
{
	cv::Mat src, dst;
	src = cv::imread(fname);

	//detectLane(src, dst);

	cv::imshow("Original Image", src);
	//cv::imshow("Edges", edge);
	//cv::imshow("Residual Edges", edgeH);
	cv::imshow("Color Edges", dst);
	int ck = cv::waitKey(0);
	if (char(ck) == 'q')
	{
		cv::destroyAllWindows();
	}
}

void testVideo(const std::string &fname)
{
	cv::VideoCapture cap;
	cap.open(fname);

	std::string winN = "Lane Detection";
	cv::namedWindow(winN, cv::WINDOW_AUTOSIZE);
	cv::Mat src, dst,dst1, tm_vp;
	int ct = 0;
	while (cap.read(src))
	{
		if (ct == 0)
		{
			constructPerspectiveMapping(tm_vp, src.size());
		}
		detectLane(src,tm_vp,dst,dst1);
		cv::imshow(winN, dst);
		cv::imshow("test", dst1);
		int kc= cv::waitKey(0);
		if (char(kc) == 'q')
		{
			break;
		}
		ct++;
	}
	cv::destroyAllWindows();
}


int main()
{
	//testImage("./test_images/test2.jpg");
	testVideo("./challenge_video.mp4");

	std::system("pause");
    return 0;
}

