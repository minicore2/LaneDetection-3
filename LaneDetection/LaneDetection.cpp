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
	ld.defineROI(0.6, 0.85, 0.47, 0.53, 0.2, 0.83); // hard video
		
	std::string winN = "Lane Detection";
	cv::namedWindow(winN, cv::WINDOW_AUTOSIZE);
	cv::Mat src;
	cv::Mat gndView, gndMarker, gndGray, srcBnd, dst;
	int ct = 0;
	while (cap.read(src))
	{
		if (ct == 0)
		{
			// normal video settings
			ld.constructPerspectiveMapping(src.size(),
		    		0.0, 0, 1.0, 0, 2.2*CV_PI / 180, 0); 
			ld.constructLUT(src.size(), 400, 0.15, 0.045); 
			
			
			// hard video settings
			/*ld.constructPerspectiveMapping(src.size(),
				0, 0, 1, 0, 3*CV_PI / 180, 0);
			ld.constructLUT(src.size(), 400, 0.15, 0.02);  // hard video
			*/

			ld.initKF(ld.mXMap.size());
		}
		if (ct == 6)
		{
			double haha = 1;
		}
		std::vector<cv::Point2d> lpts, rpts;
		ld.detectLane(src,srcBnd,gndGray,gndView,gndMarker,dst);
		cv::imshow(winN, gndView);
		cv::imshow("Markers", gndMarker);
		cv::imshow("Ground", gndGray);
		cv::imshow("Bound", srcBnd);
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

void testAutoConstrast(const std::string &fname)
{
	cv::Mat src,gray, dst;
	src = cv::imread(fname);
	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	LaneDetector::autoContrast(src, dst);

	cv::imshow("Original Image", src);
	cv::imshow("Balanced Image", dst);
	int ck = cv::waitKey(0);
	if (char(ck) == 'q')
	{
		cv::destroyAllWindows();
	}
}

int main()
{
	//testImage("./test_images/test2.jpg");
	testVideo("./challenge_video.mp4");
	//testVideo("./project_video.mp4");
	//testAutoConstrast("./test_images/test2.jpg");
	//testVideo("ZionScenicDrive.mp4");
	//testVideo("driving_at_night.mp4");

	std::system("pause");
    return 0;
}

