#include "LaneDetector.h"

bool findInnerEdgesFcn(const cv::Mat & gray, std::vector<cv::Point>& lpts, std::vector<cv::Point>& rpts)
{
	lpts.clear();
	rpts.clear();
	for (int i = gray.rows - 1; i >= 0; --i)
	{
		cv::Point lpt, rpt;
		lpt.y = i;
		rpt.y = i;
		bool rFound = false, lFound = false;
		for (int j = 0; j < gray.cols / 2 - 1; ++j)
		{
			int jr = gray.cols / 2 + j;
			int jl = gray.cols / 2 - j;
			if (gray.at<uchar>(i, jr) > 200 && !rFound)
			{
				rpt.x = jr;
				rFound = true;
			}
			if (gray.at<uchar>(i, jl) > 200 && !lFound)
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
	return true;

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
	p2.x = cvRound(x0 + (ymax - y0) / vy*vx);
}

void getGroundImageFromVC(const cv::Mat &src, const cv::Mat &tm_vp,
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

			x_v.at<double>(0, 0) = xmap.at<double>(i, j);
			x_v.at<double>(1, 0) = ymap.at<double>(i, j);
			x_v.at<double>(2, 0) = zmap.at<double>(i, j);
			x_p = tm_vp*x_v;
			x_p = x_p / x_p.at<double>(2, 0); // normalize
			int u = int(x_p.at<double>(0, 0));
			int v = int(x_p.at<double>(1, 0));
			if (u >= 0 && u<src.cols &&
				v >= 0 && v<src.rows)
			{
				img.at<cv::Vec3b>(i, j) = src.at<cv::Vec3b>(v, u);
			}
		}
	}
}



void LaneDetector::constructPerspectiveMapping(cv::Size imgSize)
{
	cv::Mat Mext;
	getCameraExtrinsicMatrix3x4(0, 0, 1, 0, 3.0*CV_PI / 180, 0,
		Mext);
	cv::Mat Mint;
	double tx_cp = imgSize.width / 2;
	double ty_cp = imgSize.height / 2;
	
	double resX = imgSize.width, resY = imgSize.height;
	getCameraIntrinsicMatrix3x3(mFocalX, mFocalY, mCCDX, mCCDY,
		resX, resY, Mint);
	mTM_vp = Mint*Mext;
}

void LaneDetector::constructLUT(cv::Size imgSize,
	int width, double mppx, double mppy)
{
	// save variable
	mMPPx = mppx; mMPPy = mppy;
	
	// setup output image
	int xsize = width;
	int ysize = int(xsize*double(imgSize.height) / imgSize.width);

	// initialize LUTs
	mXMap = cv::Mat::zeros(cv::Size(xsize, ysize), CV_64FC1);
	mYMap = cv::Mat::zeros(cv::Size(xsize, ysize), CV_64FC1);
	mZMap = cv::Mat::zeros(cv::Size(xsize, ysize), CV_64FC1);
	mLUT_u = cv::Mat::zeros(cv::Size(xsize, ysize), CV_32SC1);
	mLUT_v = cv::Mat::zeros(cv::Size(xsize, ysize), CV_32SC1);

	//double mppx = 0.2; // meter per pixel
	//double mppy = 0.1;

	// back projection (parallel projection)
	cv::Mat srcPt = cv::Mat::ones(3, 1, CV_64FC1);
	cv::Mat gndPt = cv::Mat::ones(4, 1, CV_64FC1);
	std::vector<cv::Mat> gndMap;
	for (int i = 0; i < ysize; ++i)
	{
		for (int j = 0; j < xsize; ++j)
		{
			gndPt.at<double>(0, 0) = (xsize - i)*mppx;
			gndPt.at<double>(1, 0) = (ysize*0.5 - j)*mppy;
			gndPt.at<double>(2, 0) = 0;
			srcPt = mTM_vp* gndPt;
			srcPt = srcPt / srcPt.at<double>(2, 0); // normalize
			int u = int(srcPt.at<double>(0, 0));
			int v = int(srcPt.at<double>(1, 0));
			// fill in the LUTs
			mXMap.at<double>(i, j) = gndPt.at<double>(0, 0);
			mYMap.at<double>(i, j) = gndPt.at<double>(1, 0);
			mZMap.at<double>(i, j) = gndPt.at<double>(2, 0);
			mLUT_u.at<int>(i, j) = u;
			mLUT_v.at<int>(i, j) = v;
		}
	}
}

void LaneDetector::getGroundImage(const cv::Mat & src, cv::Mat & imgG)
{
	// setup output image
	int xsize = mXMap.cols;
	int ysize = mXMap.rows;
	imgG = cv::Mat::zeros(cv::Size(xsize, ysize), CV_8UC1);
	
	// back projection (parallel projection)
	for (int i = 0; i < ysize; ++i)
	{
		for (int j = 0; j < xsize; ++j)
		{
			int u = mLUT_u.at<int>(i, j);
			int v = mLUT_v.at<int>(i, j);
			if (u >= 0 && u < src.cols &&
				v >= 0 && v < src.rows)
			{
				imgG.at<uchar>(i, j) = src.at<uchar>(v, u);
			}
		}
	}
}

bool LaneDetector::findInnerEdges(const cv::Mat & gray, 
	std::vector<cv::Point> &lpts, std::vector<cv::Point> &rpts)
{
	// convert to mat format
	cv::Mat_<cv::Vec2f> points;
	for (int i = 0; i < gray.rows; ++i)
	{
		for (int j = 0; j < gray.cols; ++j)
		{
			if (gray.at<uchar>(i, j) > 200)
			{
				cv::Vec2f pt;
				pt[0] = j;
				pt[1] = i;
				points.push_back(pt);
			}
		}
	}
	if (points.rows == 0)
	{
		return false;
	}
		
	// make points for km
	cv::Mat pointskm;
	points.copyTo(pointskm);
	for (int i = 0; i < pointskm.rows; ++i)
	{
		pointskm.at<cv::Point2f>(i).y = 0;
	}
	
	// kmean now
	cv::Mat labels, centers;
	int nAttemp = 3;
	int nCluster = 3;
	cv::kmeans(pointskm, nCluster, labels,
		cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
			10, 1.0), nAttemp, cv::KMEANS_PP_CENTERS, centers);

	// seperate points now and save the extreme values
	std::vector<cv::Mat_<cv::Vec2b>> pointSets(nCluster);
	std::vector<float> xMin(nCluster, 1e10),
		xMax(nCluster, 0);
	for (int i = 0; i < points.rows; ++i)
	{
		int iCluster = labels.at<int>(i);
		cv::Vec2b pt = points.at<cv::Vec2f>(i);
		pointSets[iCluster].push_back(pt);
		
		if (pt[0] < xMin[iCluster])
		{
			xMin[iCluster] = pt[0];
		}
		if (pt[0] > xMax[iCluster])
		{
			xMax[iCluster] = pt[0];
		}
	}
		
	// get centerX and ranges
	std::vector<std::pair<int,float>> centerX;
	for (int i = 0; i < centers.rows; ++i)
	{
		std::pair<int, float> pp;
		pp.first = i;
		pp.second = centers.at<float>(i, 0);
		if (xMax[i] - xMin[i] < 20)
		{
			centerX.push_back(pp);
		}		
	}
	std::sort(centerX.begin(), centerX.end(),
		[](auto p1, auto p2) {return p1.second < p2.second; });
	
	lpts.clear();
	rpts.clear();

	if (centerX.size() < 2)
	{
		return false;
	}
	
	int lidx = centerX.front().first;
	int ridx = centerX.back().first;
	lpts = pointSets[lidx];
	rpts = pointSets[ridx];

	
	/*
	//std::cout << "centers" << centers << "\n";
	//std::cout << "labels" << labels << "\n";
	bool rightFirst = (centers.at<float>(0,0) >= centers.at<float>(1,0));
	
	for (int i = 0; i < points.rows; ++i)
	{
		int iCluster = labels.at<int>(i);
		if (iCluster == 0)
		{
			if (rightFirst)
			{
				rpts.push_back(points.at<cv::Point2f>(i));
			}
			else
			{
				lpts.push_back(points.at<cv::Point2f>(i));
			}
		}
		else
		{
			if (rightFirst)
			{
				lpts.push_back(points.at<cv::Point2f>(i));
			}
			else
			{
				rpts.push_back(points.at<cv::Point2f>(i));
			}
		}
	}*/
	return true;
}

void LaneDetector::detectLane(const cv::Mat & src, cv::Mat & dst, cv::Mat & dst1)
{
	// blur
	//cv::Mat blurImg;
	//cv::blur(src, blurImg, cv::Size(5, 5), cv::Point(3, 3));

	// enhance constrast
	cv::Mat srcEh = src.clone();
	srcEh.forEach<cv::Point3_<uint8_t>>(changeConstrast());

	// color thresholding
	cv::Mat maskColor;
	colorThresholding(srcEh, maskColor);

	// gray
	cv::Mat gray, grayC;
	cv::cvtColor(srcEh, gray, cv::COLOR_BGR2GRAY);
	cv::bitwise_and(gray, gray, grayC, maskColor);

	// bound
	cv::Mat grayCB;
	defineROI(grayC, grayCB);

	// project to ground image
	cv::Mat grayG;
	getGroundImage(grayCB, grayG);

	// edge thresholding
	cv::Mat gradX, gradY, absGradX, absGradY, edgeG;

	//cv::Mat kernelX= cv::getGaussianKernel(3, 0.2);
	cv::Mat kernelX = (cv::Mat_<double>(7, 1) << -0.04664411095,
		-0.351327004776,
		0.0,
		0.865324982453,
		0.0,
		-0.351327004776,
		-0.04664411095);
	cv::Mat kernelY = (cv::Mat_<double>(5, 1) << 0.118364944627,
		0.321749278106,
		0.874605215994,
		0.321749278106,
		0.118364944627);

	//std::cout << "kernelX= " << kernelX << "\n";
	//std::cout << "kernelY= " << kernelY << "\n";

	cv::sepFilter2D(grayG, edgeG, CV_64F, kernelX, kernelY);

	cv::convertScaleAbs(edgeG, edgeG);
	double min, max;
	cv::minMaxLoc(edgeG, &min, &max);
	cv::threshold(edgeG, edgeG, max*0.5, 255, cv::THRESH_BINARY);

	// find inner edges
	std::vector<cv::Point> lpts, rpts;
	if (findInnerEdges(edgeG, lpts, rpts))
	{
		//  debugging codes
		cv::Mat ptImgL = cv::Mat::zeros(grayG.size(), CV_8UC1);
		cv::Mat ptImgR = cv::Mat::zeros(grayG.size(), CV_8UC1);
		for (auto pp : rpts)
		{
			cv::circle(ptImgR, pp, 3, cv::Scalar(255));
			cv::circle(grayG, pp, 3, cv::Scalar(255));
		}
		for (auto pp : lpts)
		{
			cv::circle(ptImgL, pp, 3, cv::Scalar(255));
			cv::circle(grayG, pp, 5, cv::Scalar(255));
		}
	}

	// find line
	cv::Vec3d rline;
	if (fitLine(rpts, rline))
	{
		cv::Point pt1, pt2;
		pt1.y = 0; pt2.y = grayG.rows;
		pt1.x = -rline[1] / rline[0] * pt1.y - rline[2] / rline[0];
		pt2.x = -rline[1] / rline[0] * pt2.y - rline[2] / rline[0];

		cv::line(grayG, pt1, pt2, cv::Scalar(255));
	}

	cv::Vec3d lline;
	if (fitLine(lpts, lline))
	{
		cv::Point pt1, pt2;
		pt1.y = 0; pt2.y = grayG.rows;
		pt1.x = -lline[1] / lline[0] * pt1.y - lline[2] / lline[0];
		pt2.x = -lline[1] / lline[0] * pt2.y - lline[2] / lline[0];

		cv::line(grayG, pt1, pt2, cv::Scalar(255),2);
	}

	/*
	// bound the image
	cv::Mat maskBnd;
	maskEdgeByPolygon(maskColor, maskBnd);


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
	/*std::vector<cv::Vec4i> linesR, linesL;
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
	line(srcEh, pt1, pt2, cv::Scalar(0, 0, 255), 3, CV_AA);
	}
	for (size_t i = 0; i < linesL.size(); i++)
	{
	cv::Point pt1, pt2;
	pt1.x = linesL[i][0];  pt1.y = linesL[i][1];
	pt2.x = linesL[i][2];  pt2.y = linesL[i][3];
	lptsHT.push_back(pt1);
	lptsHT.push_back(pt2);
	line(srcEh, pt1, pt2, cv::Scalar(0, 255, 255), 3, CV_AA);
	}
	*/

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

	dst = edgeG.clone();
	dst1 = grayG.clone();
}

void LaneDetector::defineROI(const cv::Mat & gray, cv::Mat & dst)
{
	cv::Mat mask;
	mask = cv::Mat::zeros(gray.size(), gray.type());

	cv::Point pts[1][4];
	int cols = gray.cols;
	int rows = gray.rows;
	pts[0][0] = (cv::Point(cols*0.05, rows*0.9));
	pts[0][1] = (cv::Point(cols*0.95, rows*0.9));
	pts[0][2] = (cv::Point(cols*0.55, 0.65*rows));
	pts[0][3] = (cv::Point(cols*0.45, 0.65*rows));

	int npts[] = { 4 };
	const cv::Point* ppt[1] = { pts[0] };
	cv::fillPoly(mask, ppt, npts, 1, cv::Scalar(255));
	cv::bitwise_and(gray, gray, dst, mask);
}

void LaneDetector::colorThresholding(const cv::Mat & src, cv::Mat & maskOut)
{
	// convert to hsv
	cv::Mat edgeH;
	cv::cvtColor(src, edgeH, cv::COLOR_BGR2HSV);

	// color thresholding
	cv::Scalar lby(14, 178, 127); // yellow
	cv::Scalar uby(57, 255, 255);
	cv::Scalar lbw(0, 0, 200); // white
	cv::Scalar ubw(255, 255, 255);
	
	cv::Mat edgeY, edgeW;//, edgeG;
	cv::inRange(edgeH, lby, uby, edgeY);
	cv::inRange(edgeH, lbw, ubw, edgeW);

	cv::bitwise_or(edgeY, edgeW, maskOut);
}

bool LaneDetector::fitLine(const std::vector<cv::Point> pts, cv::Vec3d & line)
{
	int iter = 0; int iterMax = 100;
	int npt = pts.size();
	if (npt < 2)
	{
		return false;
	}
	int nrpt = 2;

	// build original point matrix 
	cv::Mat ptMat0(npt, 3, CV_64FC1);
	for (int i = 0; i < npt; ++i)
	{
		ptMat0.at<double>(i, 0) = pts[i].x;
		ptMat0.at<double>(i, 1) = pts[i].y;
		ptMat0.at<double>(i, 2) = 1;
	}

	std::vector<cv::Point> randomPts;
	randomPts = pts;
	double costMin = 1e20, aMin, bMin, cMin;
	while (iter < iterMax)
	{
		// select random points
		if (npt > 2)
		{
			nrpt = std::min(npt - 1, 4);
		}
		std::random_shuffle(randomPts.begin(), randomPts.end());

		// least square
		cv::Mat ptMat(nrpt, 3, CV_64FC1);
		for (int i = 0; i < nrpt; ++i)
		{
			ptMat.at<double>(i, 0) = randomPts[i].x;
			ptMat.at<double>(i, 1) = randomPts[i].y;
			ptMat.at<double>(i, 2) = 1;
		}
		cv::Mat w, u, vt;
		cv::SVDecomp(ptMat, w, u, vt);
		cv::Mat coef = (vt.row(vt.rows - 1)).t();
		double a = coef.at<double>(0, 0);
		double b = coef.at<double>(1, 0);
		double c = coef.at<double>(2, 0);

		//std::cout << "coef= " << coef << "\n";

		// cost function 
		cv::Mat errVec = ptMat0*coef;
		//std::cout << "ptMat0= " << ptMat0 << "\n";
		//std::cout << "errVec= " << errVec << "\n";
		cv::Scalar err = cv::sum(cv::abs(errVec));
		double cost = err.val[0] + 1 * std::abs(b / a);
		
		if (cost < costMin)
		{
			costMin = cost;
			aMin = a; bMin = b; cMin = c;
		}

		iter++;
	}
	// create end points of the line
	line[0] = aMin; line[1] = bMin, line[2] = cMin;
}
