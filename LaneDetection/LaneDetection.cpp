// LaneDetection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2\opencv.hpp>

void findInnerEdges(const cv::Mat &img,
	std::vector<cv::Point> &lpts,
	std::vector<cv::Point> &rpts)
{
	lpts.clear();
	rpts.clear();
	for (int i = img.rows - 1; i >= 0; --i)
	{
		cv::Point lpt, rpt;
		lpt.y = i;
		rpt.y = i;
		bool rFound= false, lFound=false;
		for (int j = 0; j < img.cols / 2 - 1; ++j)
		{
			int jr = img.cols / 2 + j;
			int jl = img.cols / 2 - j; 
			if (img.at<uchar>(i,jr)> 200  && !rFound)
			{
				rpt.x = jr;
				rFound= true;
			}
			if (img.at<uchar>(i, jl) > 200  && !lFound)
			{
				lpt.x = jl;
				lFound = true;
			}
			if (rFound && lFound)
			{
				break;
			}
		}
		if (rFound)
		{
			rpts.push_back(rpt);
		}
		if (lFound)
		{
			lpts.push_back(lpt);
		}
	}
}
void getLineEndPoints(const cv::Vec4f &line, 
	const std::vector<cv::Point> &pts, int ymax, 
	cv::Point &p1, cv::Point &p2)
{
	int y = pts.back().y;
	float vx = line[0];
	float vy = line[1];
	float x0 = line[2];
	float y0 = line[3];
	int x = cvRound(x0 + (y - y0) / vy*vx);
	p1.x = x; p1.y = y;
	p2.y = ymax;
	p2.x= cvRound(x0 + (ymax - y0) / vy*vx);
}

void maskEdgeByPolygon(const cv::Mat &grayEdge, cv::Mat &dst)
{
	cv::Mat mask;
	mask = cv::Mat::zeros(grayEdge.size(), grayEdge.type());

	cv::Point pts[1][4];
	int cols = grayEdge.cols;
	int rows = grayEdge.rows;
	pts[0][0] = (cv::Point(cols*0.05, rows*0.9));
	pts[0][1] = (cv::Point(cols*0.95, rows*0.9));
	pts[0][2] = (cv::Point(cols*0.45, 0.5*rows));
	pts[0][3] = (cv::Point(cols*0.55, 0.5*rows));

	int npts[] = { 4 };
	const cv::Point* ppt[1] = { pts[0] };
	cv::fillPoly(mask, ppt, npts, 1, cv::Scalar(255));
	cv::bitwise_and(grayEdge, mask, dst);
}

void colorThresholding(const cv::Mat &src, cv::Mat &maskOut)
{
	// convert to hsv
	cv::Mat edgeH;
	cv::cvtColor(src, edgeH, cv::COLOR_BGR2HSV);

	// color thresholding
	cv::Scalar lb(10, 105, 200); //30,145,255 yellow
	cv::Scalar ub(40, 175, 255);
	cv::Scalar lbw(10, 0, 200); //32,8,255 white
	cv::Scalar ubw(40, 40, 255);
	cv::Scalar lbg(160, 0, 140); //188,7,163 gray
	cv::Scalar ubg(220, 40, 180);
	cv::Mat edgeY, edgeW, edgeG;
	cv::inRange(edgeH, lb, ub, edgeY);
	cv::inRange(edgeH, lbw, ubw, edgeW);
	cv::inRange(edgeH, lbg, ubg, edgeG);
	cv::bitwise_or(edgeY, edgeW, maskOut);
	cv::bitwise_or(maskOut, edgeG, maskOut);
}

void detectLane(const cv::Mat &src, cv::Mat &dst)
{
	// blur
	//cv::Mat blurImg;
	//cv::blur(src, blurImg, cv::Size(5, 5), cv::Point(3, 3));

	// color thresholding
	//cv::Mat colorMask, colorImg;
	//colorThresholding(src, colorMask);
	//cv::bitwise_and(src, src, colorImg, colorMask);

	// convert to gray
	//cv::Mat gray;
	//cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

	cv::Mat hsv;
	cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

	// canny edge
	cv::Mat edge;
	cv::Canny(src, edge, 100, 255);

	// mask
	cv::Mat edgeB;
	maskEdgeByPolygon(edge, edgeB);
	
	// dilate
	cv::Mat elem = cv::getStructuringElement(cv::MORPH_ELLIPSE,
		cv::Size(11, 11), cv::Point(5, 5));
	cv::dilate(edgeB, edgeB, elem);

	// coloring edge
	//cv::Mat edgeC;
	//cv::bitwise_and(src, src, edgeC, edgeB);

	// color thresholding
	//cv::Mat edgeMask;
	//colorThresholding(edgeC, edgeMask);
	
	// find inner edges
	std::vector<cv::Point> lpts, rpts;
	findInnerEdges(edgeB, lpts, rpts);
	/* debugging codes
	for (auto p : lpts)
	{
		std::cout << "left points: " << p << "\n";
	}
	for (auto p : rpts)
	{
		std::cout << "right points: " << p << "\n";
	}*/

	dst = src.clone();
	if (lpts.size() > 1 && rpts.size() > 1)
	{
		cv::Vec4f lLine, rLine;
		cv::fitLine(lpts, lLine, CV_DIST_L2, 0, 0.01, 0.01);
		cv::fitLine(rpts, rLine, CV_DIST_L2, 0, 0.01, 0.01);

		// get inner edges end points
		cv::Point rp1, rp2, lp1, lp2;
		getLineEndPoints(lLine, lpts, edgeB.rows - 1, lp1, lp2);
		getLineEndPoints(rLine, rpts, edgeB.rows - 1, rp1, rp2);

		// draw lines
		cv::Mat edgeLine = cv::Mat::zeros(edgeB.size(), CV_8UC3);
		line(edgeLine, lp1, lp2, cv::Scalar(0, 0, 255), 10, 8);
		line(edgeLine, rp1, rp2, cv::Scalar(0, 0, 255), 10, 8);

		/*  debugging codes
		for (auto pp : rpts)
		{
			cv::circle(edgeLine, pp, 3, cv::Scalar(0, 0, 255));
		}
		for (auto pp : lpts)
		{
			cv::circle(edgeLine, pp, 3, cv::Scalar(0, 230, 255));
		}
		*/

		// blend
		cv::addWeighted(edgeLine, 0.5, src, 1.0, 0, dst);
	}
	dst = hsv.clone();
}

void testImage(const std::string &fname)
{
	cv::Mat src, dst;
	src = cv::imread(fname);

	detectLane(src, dst);

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
	cv::Mat src, dst;
	while (cap.read(src))
	{
		detectLane(src, dst);
		cv::imshow(winN, dst);
		int kc= cv::waitKey(0);
		if (char(kc) == 'q')
		{
			break;
		}
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

