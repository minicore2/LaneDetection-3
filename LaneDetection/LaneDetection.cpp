// LaneDetection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "LaneDetector.h"


void testImage(const std::string &fname)
{
	/*cv::Mat src, dst;
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
	}*/
}

void testVideo(const std::string &fname)
{
	cv::VideoCapture cap;
	cap.open(fname);

	LaneDetector ld;
	
	std::string winN = "Lane Detection";
	cv::namedWindow(winN, cv::WINDOW_AUTOSIZE);
	cv::Mat src;
	cv::Mat gndView, gndMarker, gndGray, bndGray, dst;
	int ct = 0;
	while (cap.read(src))
	{
		if (ct == 0)
		{
			ld.constructPerspectiveMapping(src.size(),
				0, 0, 1, 0, 2.2*CV_PI / 180, 0);
			//ld.constructLUT(src.size(), 400, 0.05, 0.025);
			ld.constructLUT(src.size(), 400, 0.15, 0.045);
			//ld.constructLUT(src.size(), 400, 0.2, 0.06);
			ld.initKF(ld.mXMap.size());
		}
		if (ct == 6)
		{
			double haha = 1;
		}
		ld.detectLane(src,gndView,gndMarker,gndGray,bndGray,dst);
		cv::imshow(winN, gndView);
		cv::imshow("Markers", gndMarker);
		cv::imshow("Ground", gndGray);
		cv::imshow("Bound", bndGray);
		cv::imshow("Overlay", dst);
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
	//testVideo("./challenge_video.mp4");
	testVideo("./project_video.mp4");

	std::system("pause");
    return 0;
}

