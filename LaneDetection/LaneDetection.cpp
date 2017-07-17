// LaneDetection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include <expvideo\CameraUtility.h>

struct changeConstrast
{
	void operator() (cv::Point3_<uint8_t> &pixel, const int *position) const
	{
		double alpha = 1.5;
		int beta = -100;
		pixel.x = cv::saturate_cast<uchar> (pixel.x*alpha + beta);
		pixel.y = cv::saturate_cast<uchar> (pixel.y*alpha + beta);
		pixel.z = cv::saturate_cast<uchar> (pixel.z*alpha + beta);
	}
};


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
	pts[0][2] = (cv::Point(cols*0.45, 0.65*rows));
	pts[0][3] = (cv::Point(cols*0.55, 0.65*rows));

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
	cv::Scalar lby(14, 178, 127); //30,145,255 yellow
	cv::Scalar uby(57, 255, 255);
	cv::Scalar lbw(0, 0, 200); //32,8,255 white
	cv::Scalar ubw(255, 255, 255);
	//cv::Scalar lbg(160, 0, 140); //188,7,163 gray
	//cv::Scalar ubg(220, 40, 180);
	
	cv::Mat edgeY, edgeW;//, edgeG;
	cv::inRange(edgeH, lby, uby, edgeY);
	cv::inRange(edgeH, lbw, ubw, edgeW);

	//cv::inRange(edgeH, lbw, ubw, edgeW);
	//cv::inRange(edgeH, lbg, ubg, edgeG);
	cv::bitwise_or(edgeY, edgeW, maskOut);
	//cv::bitwise_or(maskOut, edgeG, maskOut);
}

void constructPerspectiveMapping(cv::Mat &tm_vp, cv::Size imgSize)
{
	// build rotation matrix
	cv::Mat rm_cv, rm_vc;
	rotzyx(-90 * CV_PI / 180, 0, -90 * CV_PI / 180, rm_cv);
	rm_vc = rm_cv.t();
	// translation part
	cv::Mat t_cv= cv::Mat::zeros(3,1,CV_64FC1);
	cv::Mat t_vc;
	t_cv.at<double>(0, 0) = 0;
	t_cv.at<double>(1, 0) = 0;
	t_cv.at<double>(2, 0) = 1;
	t_vc = -rm_vc*t_cv;

	// build 3x4 transformation matrix 
	cv::Mat tm_vc= cv::Mat::zeros(3,4, CV_64FC1);
	// fill in the rotation part
	cv::Rect roi(0, 0, 3, 3);	
	rm_vc.copyTo(tm_vc(roi));
	tm_vc.at<double>(0, 3) = t_vc.at<double>(0, 0);
	tm_vc.at<double>(1, 3) = t_vc.at<double>(1, 0);
	tm_vc.at<double>(2, 3) = t_vc.at<double>(2, 0);

	// build intrinsic matrix
	double tx_cp = imgSize.width/2;
	double ty_cp = imgSize.height/2;
	double focal = 1;  // 10 cm
	double pixelPerM = 2000; // 10 pixel per meter
	cv::Mat km = cv::Mat::eye(3, 3, CV_64FC1);
	km.at<double>(0, 0) = pixelPerM*focal;
	km.at<double>(1, 1) = pixelPerM*focal;
	km.at<double>(0, 2) = tx_cp;
	km.at<double>(1, 2) = ty_cp;
	
	// assemble output transformation matrix
	tm_vp = km*tm_vc;
}

void getGroundImage(const cv::Mat &src, const cv::Mat &tm_vp,
	cv::Mat &img)
{
	Plane3D gp;
	gp.D = 0;
	gp.setNormal(0, 0, 1);
	VirtualCamera vc{ 0,0,10,0,-90 * CV_PI / 180,0 };
	vc.fovAngle = 135 * CV_PI / 180;
	vc.resolutionX = 400;
	vc.aspect = 1;
	
	cv::Mat xmap, ymap, zmap;
	vc.hitPlane(gp, xmap, ymap, zmap);
	// allocate output image size
	img = cv::Mat::zeros(xmap.size(), CV_8UC3);

	cv::Mat x_v = cv::Mat::ones(4, 1, CV_64FC1);
	cv::Mat x_p;
	// fill in colors
	for (int i = 0; i < img.rows; ++i)
	{
		for (int j = 0; j < img.cols; ++j)
		{
			
			x_v.at<double>(0,0) = xmap.at<double>(i, j);
			x_v.at<double>(1, 0) = ymap.at<double>(i, j);
			x_v.at<double>(2, 0) = zmap.at<double>(i, j);
			x_p= tm_vp*x_v;
			x_p = x_p / x_p.at<double>(2, 0); // normalize
			int u = int(x_p.at<double>(0, 0));
			int v = int(x_p.at<double>(1, 0));
			if (u>=0 && u<src.cols &&
				v>=0 && v<src.rows)
			{
				img.at<cv::Vec3b>(i, j) = src.at<cv::Vec3b>(v, u);
			}
		}
	}
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

	// ehnace constrast
	cv::Mat srcEh = src.clone();
	srcEh.forEach<cv::Point3_<uint8_t>>(changeConstrast());
	
	cv:: Mat tm_vp;
	constructPerspectiveMapping(tm_vp, srcEh.size());

	cv::Mat srcG;
	getGroundImage(src, tm_vp, srcG);

	// gray
	cv::Mat gray;
	cv::cvtColor(srcEh, gray, cv::COLOR_BGR2GRAY);
	
	// canny edge
	cv::Mat edge;
	cv::Canny(gray, edge, 100, 255, 5);

	// dilate
	cv::Mat maskEdge;
	cv::Mat elem = cv::getStructuringElement(cv::MORPH_RECT,
		cv::Size(11, 11), cv::Point(5, 5));
	cv::dilate(edge, maskEdge, elem);
	
	// color the edge and convert color to hsv
	cv::Mat hsv;
	cv::bitwise_and(srcEh, srcEh, hsv, maskEdge);
	cv::cvtColor(hsv, hsv, cv::COLOR_BGR2HSV);

	// color thresholding
	cv::Mat maskColor;
	colorThresholding(hsv, maskColor);

	// bound the image
	cv::Mat maskBnd;
	maskEdgeByPolygon(maskColor, maskBnd);
	
	// find inner edges
	std::vector<cv::Point> lpts, rpts;
	findInnerEdges(maskBnd, lpts, rpts);
	//  debugging codes
	cv::Mat ptImgL = cv::Mat::zeros(src.size(), CV_8UC1);
	cv::Mat ptImgR = cv::Mat::zeros(src.size(), CV_8UC1);
	for (auto pp : rpts)
	{
		cv::circle(ptImgR, pp, 3, cv::Scalar(255));
		cv::circle(srcEh, pp, 3, cv::Scalar(255,0,0));
	}
	for (auto pp : lpts)
	{
		cv::circle(ptImgL, pp, 3, cv::Scalar(255));
		cv::circle(srcEh, pp, 3, cv::Scalar(0,255,255));
	}
	/* debugging codes
	for (auto p : lpts)
	{
	std::cout << "left points: " << p << "\n";
	}
	for (auto p : rpts)
	{
	std::cout << "right points: " << p << "\n";
	}*/
	
	// hough lines
	std::vector<cv::Vec4i> linesR, linesL;
	std::vector<cv::Point> lptsHT, rptsHT;
	cv::HoughLinesP(ptImgR, linesR, 10, CV_PI / 180, 50, 20, 10);
	cv::HoughLinesP(ptImgL, linesL, 10, CV_PI / 180, 50, 20, 10);
	for (size_t i = 0; i < linesR.size(); i++)
	{
		cv::Point pt1, pt2;
		pt1.x = linesR[i][0];  pt1.y = linesR[i][1];
		pt2.x = linesR[i][2];  pt2.y = linesR[i][3];
		rptsHT.push_back(pt1);
		rptsHT.push_back(pt2);
		line(srcEh, pt1,pt2, cv::Scalar(0, 0, 255), 3, CV_AA);
	}
	for (size_t i = 0; i < linesL.size(); i++)
	{
		cv::Point pt1, pt2;
		pt1.x = linesL[i][0];  pt1.y = linesL[i][1];
		pt2.x = linesL[i][2];  pt2.y = linesL[i][3];
		lptsHT.push_back(pt1);
		lptsHT.push_back(pt2);
		line(srcEh, pt1,pt2, cv::Scalar(0, 255, 255), 3, CV_AA);
	}


	// coloring edge
	//cv::Mat edgeC;
	//cv::bitwise_and(src, src, edgeC, edgeB);

	// color thresholding
	//cv::Mat edgeMask;
	//colorThresholding(edgeC, edgeMask);
		
	/*if (lptsHT.size() > 1 && rptsHT.size() > 1)
	{
		cv::Vec4f lLine, rLine;
		cv::fitLine(lptsHT, lLine, CV_DIST_L2, 0, 0.01, 0.01);
		cv::fitLine(rptsHT, rLine, CV_DIST_L2, 0, 0.01, 0.01);

		// get inner edges end points
		cv::Point rp1, rp2, lp1, lp2;
		getLineEndPoints(lLine, lptsHT, maskBnd.rows - 1, lp1, lp2);
		getLineEndPoints(rLine, rptsHT, maskBnd.rows - 1, rp1, rp2);

		// draw lines
		cv::Mat edgeLine = cv::Mat::zeros(maskBnd.size(), CV_8UC3);
		line(edgeLine, lp1, lp2, cv::Scalar(0, 0, 255), 10, 8);
		line(edgeLine, rp1, rp2, cv::Scalar(0, 0, 255), 10, 8);
		
		// blend
		cv::addWeighted(edgeLine, 0.5, srcEh, 1.0, 0, dst);
	}*/
	
	dst = srcG.clone();
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

