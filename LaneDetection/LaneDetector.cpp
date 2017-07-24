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

LaneDetector::LaneDetector()
{
	mKFL = new cv::KalmanFilter(16, 8, 0);
	mKFR = new cv::KalmanFilter(16, 8, 0);
}

LaneDetector::~LaneDetector()
{
	delete mKFL; mKFL = nullptr;
	delete mKFR; mKFR = nullptr;
}

void LaneDetector::constructPerspectiveMapping(cv::Size imgSize,
	double x, double y, double z,
	double pan, double tilt, double roll)
{
	cv::Mat Mext;
	getCameraExtrinsicMatrix3x4(x, y, z, pan, tilt, roll,Mext);
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
			gndPt.at<double>(0, 0) = (ysize - i)*mppx;
			gndPt.at<double>(1, 0) = (xsize*0.5 - j)*mppy;
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

void LaneDetector::initKF(cv::Size imgSize)
{
	// -----for left-----
	prevMeasL = cv::Mat::zeros(8, 1, CV_32FC1);
	mKFL->transitionMatrix = cv::Mat::eye(16, 16, CV_32FC1);
	// speed part
	cv::Rect roi(8,0,8,8);
	mKFL->transitionMatrix(roi)= 1*cv::Mat::eye(8, 8, CV_32FC1);

	for (int i = 0; i < 16; ++i)
	{
		mKFL->statePre.at<float>(i) = 0;
	}	
	cv::setIdentity(mKFL->measurementMatrix);
	cv::setIdentity(mKFL->processNoiseCov, cv::Scalar::all(1e-4));
	cv::setIdentity(mKFL->measurementNoiseCov, cv::Scalar::all(20));
	cv::setIdentity(mKFL->errorCovPost, cv::Scalar::all(0.1));

	// -----for right------
	prevMeasR = cv::Mat::zeros(8, 1, CV_32FC1);
	mKFR->transitionMatrix = cv::Mat::eye(16, 16, CV_32FC1);
	// speed part
	mKFR->transitionMatrix(roi) = 1*cv::Mat::eye(8, 8, CV_32FC1);

	for (int i = 0; i < 16; ++i)
	{
		mKFR->statePre.at<float>(i) = 0;
	}
	cv::setIdentity(mKFR->measurementMatrix);
	cv::setIdentity(mKFR->processNoiseCov, cv::Scalar::all(1e-4));
	cv::setIdentity(mKFR->measurementNoiseCov, cv::Scalar::all(20));
	cv::setIdentity(mKFR->errorCovPost, cv::Scalar::all(0.1));
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

bool LaneDetector::groupPoints(const cv::Mat & gray, 
	std::vector<cv::Point> &lpts, std::vector<cv::Point> &rpts)
{
	// get "on" points and convert to mat format
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
	// if no point available, return false
	if (points.rows < 3)
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
	double lineWidth = 0.4 / mMPPx; 
	for (int i = 0; i < centers.rows; ++i)
	{
		std::pair<int, float> pp;
		pp.first = i;
		pp.second = centers.at<float>(i, 0);
		if (xMax[i] - xMin[i] < lineWidth)
		{
			centerX.push_back(pp);
		}		
	}
	// ascending sort centerX
	std::sort(centerX.begin(), centerX.end(),
		[](auto p1, auto p2) {return p1.second < p2.second; });
	
	lpts.clear();
	rpts.clear();
	
	// if no clusters, return false
	if (centerX.size() < 1)
	{
		return false;
	}
	
	// check centers X distance. If too close, 
	// combine clusters
	int lidx = centerX.front().first;
	int ridx = centerX.back().first;
	float lx = centerX.front().second;
	float rx = centerX.back().second;
	if (std::abs(rx - lx) < lineWidth)
	{
		lpts = pointSets[lidx];
		lpts.insert(lpts.end(), pointSets[ridx].begin(), pointSets[ridx].end());
	}
	else
	{
		lpts = pointSets[lidx];
		rpts = pointSets[ridx];
	}
	
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

void LaneDetector::getPointsFromImage(const cv::Mat & gray,
	int uStart, int uEnd, int vStart, int vEnd,
	cv::Mat &points)
{
	cv::Mat_<cv::Vec2f> pts;
	for (int i = vStart; i <= vEnd; ++i)
	{
		for (int j = uStart; j <=uEnd; ++j)
		{
			if (gray.at<uchar>(i, j) > 230)
			{
				cv::Vec2f pt;
				pt[0]= j;
				pt[1] = i;
				pts.push_back(pt);
			}
		}
	}
	pts.copyTo(points);
}

void LaneDetector::autoContrast(const cv::Mat & src, cv::Mat & dst,
	double histClipPct)
{
	CV_Assert(src.type() == CV_8UC1 || src.type()== CV_8UC3 ||
	src.type()==CV_8UC4);

	// convert to gray
	cv::Mat gray;
	if (src.type() == CV_8UC1)
	{
		gray = src;
	}
	else if (src.type() == CV_8UC3)
	{
		cvtColor(src, gray, CV_BGR2GRAY);
	}
	else if (src.type() == CV_8UC4)
	{
		cvtColor(src, gray, CV_BGRA2GRAY);
	}

	// determine intensity range
	double minVal, maxVal;
	int histSize = 256;
	if (histClipPct == 0)
	{
		// keep full available range
		cv::minMaxLoc(gray, &minVal, &maxVal);
	}
	else
	{
		cv::Mat hist; //the grayscale histogram

		float range[] = { 0, 256 };
		const float* histRange = { range };
		bool uniform = true;
		bool accumulate = false;
		calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

		// calculate cumulative distribution from the histogram
		std::vector<float> accumulator(histSize);
		accumulator[0] = hist.at<float>(0);
		for (int i = 1; i < histSize; i++)
		{
			accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
		}

		// locate points that cuts at required value
		float max = accumulator.back();
		histClipPct *= (max / 100.0); //make percent as absolute
		histClipPct /= 2.0; // left and right wings
		// locate left cut
		minVal = 0;
		while (accumulator[minVal] < histClipPct)
			minVal++;

		// locate right cut
		maxVal = histSize - 1;
		while (accumulator[maxVal] >= (max - histClipPct))
			maxVal--;
	}

	double rangeIn = maxVal - minVal;
	int intensityMax = 255;
	double alpha = intensityMax / rangeIn;
	double beta = -minVal*alpha;
	src.convertTo(dst, -1, alpha, beta);  // using saturate_cast

	// restore alpha channel from source 
	if (dst.type() == CV_8UC4)
	{
		int from_to[] = { 3, 3 };
		cv::mixChannels(&src, 4, &dst, 1, from_to, 1);
	}

}

void BezierSpline::computeAccumulativeLength(
	const std::vector<cv::Point>& pts, std::vector<double> &sacc)
{
	size_t npt = pts.size();
	double dx, dy, ds;
	sacc.clear();
	sacc.push_back(0);  // first point has zero length
	for (size_t i = 1; i<npt; ++i)
	{
		dx = pts[i].x - pts[i - 1].x;
		dy = pts[i].y - pts[i - 1].y;
		ds = std::sqrt(dx*dx + dy*dy) + sacc[i - 1];
		sacc.push_back(ds);
	}
}

void BezierSpline::computeAccumulativeLength(const cv::Mat & pts, 
	std::vector<double> &sacc)
{
	int npt = pts.rows;
	double dx, dy, ds;
	sacc.clear();
	sacc.push_back(0);  // first point has zero length
	for (int i = 1; i<npt; ++i)
	{
		dx = pts.at<double>(i,0) - pts.at<double>(i-1, 0);
		dy = pts.at<double>(i,1) - pts.at<double>(i-1, 1);
		ds = std::sqrt(dx*dx + dy*dy) + sacc[i - 1];
		sacc.push_back(ds);
	}
}

double BezierSpline::computeSplineCurveness()
{
	double curveness = 0;
	int npt = Pmat.rows;
	double dx1, dx2, dy1, dy2, ds1,ds2;
	for (int i = 1; i < npt-1; i++)
	{
		dx1 = Pmat.at<double>(i, 0) - Pmat.at<double>(i - 1, 0);
		dy1 = Pmat.at<double>(i, 1) - Pmat.at<double>(i - 1, 1);
		dx2 = Pmat.at<double>(i+1, 0) - Pmat.at<double>(i, 0);
		dy2 = Pmat.at<double>(i+1, 1) - Pmat.at<double>(i, 1);
		ds1 = std::sqrt(dx1*dx1 + dy1*dy1);
		ds2 = std::sqrt(dx2*dx2 + dy2*dy2);

		double cth = (dx1*dx2 + dy1*dy2) / ds1 / ds2;
		curveness += cth;
	}
	// normalize
	curveness /= npt;
	return curveness;
}

void BezierSpline::interpolatePts(int npts, 
	std::vector<cv::Point>& pts)
{
	cv::Mat TmatTest;
	constructSplineParameterMatrix(npts, TmatTest);
	cv::Mat QmatTest;
	QmatTest = (TmatTest*Mmat)*Pmat;

	pts.clear();
	for (int i = 0; i < QmatTest.rows; ++i)
	{
		cv::Point pt;
		pt.x = QmatTest.at<double>(i, 0);
		pt.y = QmatTest.at<double>(i, 1);
		pts.push_back(pt);
	}
}

bool BezierSpline::fitRANSACSpline(cv::Size imgSize,
	const std::vector<cv::Point>& pts)
{
	int npt = pts.size();
	if (npt < 6)
	{
		return false;
	}
	
	// setup randomPts
	std::vector<cv::Point> randomPts;
	randomPts = pts;
	
	// prepare for loop
	cv::Mat PmatMax;
	int iter = 0, iterMax = 50;
	double costMax = -1e20;
	while (iter < iterMax)
	{
		// select random points
		int nrpt= 5;
		
		std::random_shuffle(randomPts.begin(), randomPts.end());
		
		// sort by y descending
		std::vector<cv::Point> rpts(randomPts.begin(), randomPts.begin() + nrpt);
		std::sort(rpts.begin(), rpts.end(),
			[](auto pt1, auto pt2) {return pt1.y > pt2.y;});

		// fit now
		if (!fit(rpts))
		{
			iter++;
			continue;
		}
				
		// cost function (long and less curvy)
		// get points
		std::vector<cv::Point> ipts;
		interpolatePts(50, ipts);
		
		std::vector<double> saccTest;
		computeAccumulativeLength(ipts, saccTest);
		double curveLen = saccTest.back() / imgSize.height -1;
		double curveness = computeSplineCurveness();

		double cost = 1*curveLen + 1*curveness; // to maximize
		//std::cout << "curveness= " << curveness << "\n";
		if (cost > costMax && curveness >0.4 )
		{
			costMax = cost;
			for (int i = 0; i < Pmat.rows; ++i)
			{
				cv::Point pt;
				pt.x = Pmat.at<double>(i, 0);
				pt.y = Pmat.at<double>(i, 1);
				PmatMax = Pmat.clone();
			}
		}

		iter++;
	}
	// update control point matrix
	PmatMax.copyTo(this->Pmat);	
	if (Pmat.rows > 0)
	{
		return true;
	}
	else
	{
		return false;
	}
}

void LaneDetector::getVerticalScannedPoints(const cv::Mat & gray, 
	int nDivX_2, int nDivY, 
	std::vector<cv::Point> &pts, bool left)
{
	pts.clear();
	int xsize = gray.cols;
	int ysize = gray.rows;
	int dpx = (xsize/2) / nDivX_2;
	int dpy = ysize / nDivY;
	int minPtCount = 1;
		
	for (int iy = 0; iy < nDivY; ++iy)
	{
		// y need to be reversed
		int vStart = (ysize - 1) - iy*(dpy);
		int vEnd = (ysize - 1) - ((iy + 1)*(dpy)-1);
				
		// loop left
		int uStart, uEnd;
		for (int ix = 0; ix < nDivX_2; ++ix)
		{
			if (left)
			{
				uStart = (xsize / 2 - 1) - ((ix + 1)*(dpx)-1);
				uEnd = (xsize / 2 - 1) - ix*(dpx);
			}
			else
			{
				uStart = (xsize / 2) + ix*(dpx);
				uEnd = (xsize / 2) + ((ix + 1)*(dpx))-1;
			}
			
			cv::Mat ipts;
			getPointsFromImage(gray, uStart, uEnd, vEnd, vStart, ipts);
			if (ipts.rows >= minPtCount)
			{
				cv::Scalar meanV = cv::mean(ipts);
				cv::Point pt;
				pt.x = meanV.val[0];
				pt.y = meanV.val[1];
				pts.push_back(pt);
				break;
			}
		}
	}
}

void LaneDetector::findLaneByKFOld(const cv::Mat & gray,
	std::vector<cv::Point> &lanePts, bool left)
{
	int xsize = gray.cols;
	int ysize = gray.rows;

	int nDivY = 6;
	int dpx = 0.4 / mMPPy; // sliding window width in x dir 

	cv::Mat predicted, estimated;
	cv::Mat_<float> measurement(1, 1); measurement(0) = 0;

	lanePts.clear();
	for (int iy = 0; iy < nDivY; ++iy)
	{
		// KF prediction
		predicted = (left) ? mKFL->predict() : mKFR->predict();

		// measurement
		int nPtMax = 0;
		std::vector<float> xValues;
		float measX = predicted.at<float>(0);
		// y need to be reversed
		int vStart = (ysize - 1) - iy*(ysize / nDivY);
		int vEnd = (ysize - 1) - ((iy + 1)*(ysize / nDivY) - 1);
		float measY = (vStart + vEnd) / 2;
		for (int ix = 0; ix < xsize / 2 - dpx; ++ix)
		{
			int uStart, uEnd;
			if (left)
			{
				uStart = ix;
				uEnd = ix + dpx;
			}
			else
			{
				uStart = ix + xsize / 2;
				uEnd = ix + dpx + xsize / 2;
			}

			cv::Mat ipts;
			getPointsFromImage(gray, uStart, uEnd, vEnd, vStart, ipts);
			if (ipts.rows > nPtMax)
			{
				nPtMax = ipts.rows;
				cv::Scalar meanV = cv::mean(ipts);
				measX = meanV.val[0];
				measY = meanV.val[1];
			}
		}
		measurement(0) = measX;

		// KF update
		estimated = (left) ? mKFL->correct(measurement) :
			mKFR->correct(measurement);
		cv::Point statePt(estimated.at<float>(0), measY);

		//std::cout << "iy= " << iy << " measX= " << measX
		//	<< " measY= " << measY <<" stateX= "<<statePt <<"\n";

		lanePts.push_back(statePt);
	}
}

void LaneDetector::findLaneByKF(const cv::Mat & gray,
	std::vector<cv::Point> &lanePts, bool left)
{
	int xsize = gray.cols;
	int ysize = gray.rows;

	int nDivY = 30;
	int dpx = 0.4 / mMPPy; // sliding window width in x dir
	int nDivX_2 = (xsize / 2) / dpx;

	cv::Mat predicted, estimated;
	cv::Mat measurement= (left)? prevMeasL.clone():prevMeasR.clone();

	//std::cout << "prevMeasL= " << prevMeasL << "\n";
	//std::cout << "prevMeasR= " << prevMeasL << "\n";

	// KF prediction
	predicted = (left) ? mKFL->predict() : mKFR->predict();

	// measure
	std::vector<cv::Point> vpts;
	getVerticalScannedPoints(gray, nDivX_2, nDivY, vpts, left);
	BezierSpline bsp;
	
	bsp.fitRANSACSpline(gray.size(), vpts);
	//std::cout << "Pmat= " << bsp.Pmat << "\n";
	int nCtrlPt = bsp.Pmat.rows;
	if (nCtrlPt >0)
	{
		for (int i = 0; i < nCtrlPt; ++i)
		{
			measurement.at<float>(2 * i, 0) = float(bsp.Pmat.at<double>(i, 0));
			measurement.at<float>(2 * i+1, 0) = float(bsp.Pmat.at<double>(i, 1));
		}
	}
	
	//std::cout << "meas= " << measurement << "\n";
	// store measurement
	if (left)
	{
		measurement.copyTo(prevMeasL);
	}
	else
	{
		measurement.copyTo(prevMeasR);
	}

	// KF update
	estimated = (left) ? mKFL->correct(measurement) :
		mKFR->correct(measurement);
	// update bsp
	cv::Mat ctrlMat(4,2,CV_64FC1);
	for (int i = 0; i < 4; ++i)
	{
		ctrlMat.at<double>(i, 0) = double(estimated.at<float>(2 * i));
		ctrlMat.at<double>(i, 1) = double(estimated.at<float>(2 * i + 1));
	}
	//std::cout << "ctrlMAt= " << ctrlMat << "\n";
	bsp.Pmat = ctrlMat.clone();
	bsp.interpolatePts(20, lanePts);

}


void LaneDetector::getCamPtsFromGndImgPts(
	const std::vector<cv::Point>& gndImgPts, 
	std::vector<cv::Point>& camPts)
{
	camPts.clear();
	for (size_t i = 0; i < gndImgPts.size(); ++i)
	{
		cv::Point pt;
		int u = gndImgPts[i].x;
		int v = gndImgPts[i].y;
		if (u >= 0 && u < mLUT_u.cols &&
			v >= 0 && v < mLUT_u.rows)
		{
			pt.x = mLUT_u.at<int>(v, u);
			pt.y = mLUT_v.at<int>(v, u);
			camPts.push_back(pt);
		}
	}
}

void LaneDetector::getFilteredLines(const cv::Mat & grayG, cv::Mat & lineG)
{
	cv::Mat gradX, gradY, absGradX, absGradY, edgeG;

	// define kernels in x and y directions
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

	// sequentially filter the image
	cv::sepFilter2D(grayG, edgeG, CV_64F, kernelX, kernelY);

	// convert to abs scale
	cv::convertScaleAbs(edgeG, lineG);

	// get max value of the lineG
	double min, max;
	cv::minMaxLoc(lineG, &min, &max);

	// thresholding remove values< 0.5 max
	cv::threshold(lineG, lineG, max*0.5, 255, cv::THRESH_BINARY);
}

void LaneDetector::detectLane(const cv::Mat & src, 
	cv::Mat &gndView, cv::Mat &gndMarker, cv::Mat &gndGray,
	cv::Mat &bndGray, cv::Mat &dst)
{
	// enhance constrast
	//cv::Mat srcEh = src.clone();
	//srcEh.forEach<cv::Point3_<uint8_t>>(changeConstrast());
	cv::Mat srcEh;
	autoContrast(src, srcEh,2);
	
	// color thresholding
	cv::Mat maskColor;
	colorThresholding(srcEh, maskColor);

	// gray
	cv::Mat gray, grayC;
	cv::cvtColor(srcEh, gray, cv::COLOR_BGR2GRAY);
	cv::bitwise_and(gray, gray, grayC, maskColor);
	
	// bound
	cropToROI(grayC, bndGray);
	
	// project to ground image
	cv::Mat grayG;
	getGroundImage(bndGray, grayG);
	getGroundImage(gray, gndGray);

	cv::equalizeHist(grayG, grayG);

	// edge thresholding using custom 2D filtering
	cv::Mat lineG;
	getFilteredLines(grayG, lineG);
	
	// find lanes
	std::vector<cv::Point> lpts, rpts;
	findLaneByKF(lineG, lpts, true);  // left lane
	findLaneByKF(lineG, rpts, false); // right lane

	for (auto pp : lpts)
	{
		cv::circle(grayG, pp, 5, cv::Scalar(255));
	}
	for (auto pp : rpts)
	{
		cv::drawMarker(grayG, pp, cv::Scalar(255), cv::MARKER_CROSS,8);
	}

	// get lane points in camera image frame
	std::vector<cv::Point> lpts_p, rpts_p;
	getCamPtsFromGndImgPts(lpts, lpts_p);
	getCamPtsFromGndImgPts(rpts, rpts_p);

	// draw line
	if (lpts_p.size() >= 2)
	{
		for (int i = 0; i < lpts_p.size() - 1; ++i)
		{
			cv::line(srcEh, lpts_p[i], lpts_p[i + 1],
				cv::Scalar(0, 0, 255), 5);
		}
	}
	if (rpts_p.size() >= 2)
	{
		for (int i = 0; i < rpts_p.size() - 1; ++i)
		{
			cv::line(srcEh, rpts_p[i], rpts_p[i + 1],
				cv::Scalar(0, 255, 255), 5);
		}
	}

	gndView = lineG.clone();
	gndMarker = grayG.clone();
	dst = srcEh.clone();
}

void LaneDetector::defineROI(double nyMin, double nyMax, double nxMin_top, double nxMax_top, double nxMin_bot, double nxMax_bot)
{
	mROI_yTop = nyMin;
	mROI_yBot = nyMax;
	mROI_xminTop = nxMin_top;
	mROI_xmaxTop = nxMax_top;
	mROI_xminBot = nxMin_bot;
	mROI_xmaxBot = nxMax_bot;
}

void LaneDetector::cropToROI(const cv::Mat & gray, cv::Mat & dst)
{
	cv::Mat mask;
	mask = cv::Mat::zeros(gray.size(), gray.type());

	cv::Point pts[1][4];
	int cols = gray.cols;
	int rows = gray.rows;
	pts[0][0] = (cv::Point(cols*mROI_xminBot, rows*mROI_yBot));
	pts[0][1] = (cv::Point(cols*mROI_xmaxBot, rows*mROI_yBot));
	pts[0][2] = (cv::Point(cols*mROI_xmaxTop, rows*mROI_yTop));
	pts[0][3] = (cv::Point(cols*mROI_xminTop, rows*mROI_yTop));

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
	
	//cv::Scalar lbw(0, 0, 200); // white
	cv::Scalar lbw(0, 0, 220); // white
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

bool BezierSpline::fit(const std::vector<cv::Point>& pts)
{
	size_t npt = pts.size();
	if (npt < 4)
	{
		return false;
	}
	// construct point matrix Q
	cv::Mat Qmat(npt, 2, CV_64FC1);  // nx2 matrix
	for (size_t i = 0; i< npt; ++i)
	{
		Qmat.at<double>(i, 0) = pts[i].x;
		Qmat.at<double>(i, 1) = pts[i].y;
	}

	// construct parameter matrix T
	cv::Mat Tmat;
	constructSplineParameterMatrix(pts, Tmat);

	// build T*M matrix
	cv::Mat TMmat = Tmat*Mmat;

	// solve using normal equation
	cv::Mat mat1, mat12;  // control point matrix
	mat1 = (TMmat.t())*TMmat;
	mat12 = mat1.inv()*TMmat.t();
	Pmat = mat12*Qmat;

	return true;
}

void BezierSpline::constructSplineParameterMatrix(const std::vector<cv::Point>& pts, cv::Mat & Tmat)
{
	size_t npt = pts.size();
	// construt original parameter matrix T
	Tmat.create(npt, 4, CV_64FC1);
	// calculate parameter t for each point
	std::vector<double> sacc; // accumulated length
	computeAccumulativeLength(pts, sacc);
	double ssum = sacc.back();
	for (int i = 0; i < npt; ++i)
	{
		double t = sacc[i] / ssum;
		Tmat.at<double>(i, 0) = t*t*t;
		Tmat.at<double>(i, 1) = t*t;
		Tmat.at<double>(i, 2) = t;
		Tmat.at<double>(i, 3) = 1;
	}
}

void BezierSpline::constructSplineParameterMatrix(int npt, cv::Mat & Tmat)
{
	// construt original parameter matrix T
	Tmat.create(npt, 4, CV_64FC1);

	for (int i = 0; i < npt; ++i)
	{
		double t = double(i) / (npt - 1);
		Tmat.at<double>(i, 0) = t*t*t;
		Tmat.at<double>(i, 1) = t*t;
		Tmat.at<double>(i, 2) = t;
		Tmat.at<double>(i, 3) = 1;
	}
}
