// LaneDetection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2\opencv.hpp>

int main()
{
	cv::Mat src;
	src= cv::imread("./test_images/test2.jpg");
	
	// convert to gray
	cv::Mat gray;
	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

	// canny edge
	cv::Mat edge;
	cv::Canny(gray, edge, 100, 255);

	// mask
	cv::Mat mask, edgeB;
	mask = cv::Mat::zeros(gray.size(), gray.type());
	//cv::Rect roi(0, gray.rows*0.5, gray.cols, gray.rows*0.5);
	//mask(roi) = 255*cv::Mat::ones(cv::Size(gray.cols, gray.rows*0.5), gray.type());
	
	cv::Point pts[1][4];
	pts[0][0]= (cv::Point(gray.cols*0.05, gray.rows*0.9));
	pts[0][1]= (cv::Point(gray.cols*0.95, gray.rows*0.9));
	pts[0][2]= (cv::Point(gray.cols*0.45, 0.5*gray.rows));
	pts[0][3] = (cv::Point(gray.cols*0.55, 0.5*gray.rows));
	
	int npts[] = { 4 };
	const cv::Point* ppt[1] = { pts[0] };
	cv::fillPoly(mask, ppt, npts,1,cv::Scalar(255));
	cv::bitwise_and(edge, mask, edgeB);

	// dilate
	cv::Mat elem = cv::getStructuringElement(cv::MORPH_ELLIPSE, 
		cv::Size(11, 11), cv::Point(5, 5));
	cv::dilate(edgeB, edgeB, elem);

	// coloring edge
	cv::Mat edgeC;
	cv::bitwise_and(src,src, edgeC, edgeB);

	// convert to hsv
	cv::Mat edgeH;
	cv::cvtColor(edgeC, edgeH, cv::COLOR_BGR2HSV);

	// color thresholding
	cv::Scalar lb(10, 105, 200); //30,145,255
	cv::Scalar ub(40, 175, 255);
	cv::Scalar lbw(10, 0, 200); //32,8,255
	cv::Scalar ubw(40, 40, 255);
	cv::Scalar lbg(160, 0, 140); //188,7,163
	cv::Scalar ubg(220, 40, 180);
	cv::Mat edgeY, edgeW, edgeG;
	cv::inRange(edgeH, lb, ub, edgeY);
	cv::inRange(edgeH, lbw, ubw, edgeW);
	cv::inRange(edgeH, lbg, ubg, edgeG);
	cv::bitwise_or(edgeY, edgeW, edgeH);
	cv::bitwise_or(edgeH, edgeG, edgeH);

	/*
	// find contours
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hier;
	cv::findContours(edgeH, contours, hier, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	
	// draw contrours
	cv::Mat edgeCtr = cv::Mat::zeros(edgeH.size(), CV_8UC3);
	for (size_t i = 0; i < contours.size(); ++i)
	{
		cv::drawContours(edgeCtr, contours, i, cv::Scalar(0, 0, 255), 3, 8, hier);
	}*/

	// find line segments
	cv::Mat edgeLine = cv::Mat::zeros(edgeH.size(), CV_8UC3);
	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(edgeH, lines, 1, CV_PI / 180, 20, 100, 400);
	for (size_t i = 0; i < lines.size(); i++)
	{
		line(edgeLine, cv::Point(lines[i][0], lines[i][1]),
			cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 255), 3, 8);
	}

	cv::imshow("Original Image", src);
	cv::imshow("Edges", edge);
	cv::imshow("Residual Edges", edgeH);
	cv::imshow("Color Edges", edgeLine);
	int ck = cv::waitKey(0);
	if (char(ck) == 'q')
	{
		cv::destroyAllWindows();
	}
	std::system("pause");
    return 0;
}

