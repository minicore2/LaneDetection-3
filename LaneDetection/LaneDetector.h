#pragma once
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

class BezierSpline
{
public:
	BezierSpline() {
		// construct Bezier matrix M
		Mmat =
			(cv::Mat_<double>(4, 4) << -1, 3, -3, 1, 3, -6, 3, 0, -3, 3, 0, 0, 1, 0, 0, 0);
	};
	~BezierSpline() {};

	double computeSplineCurveness();
	void interpolatePts(int npts, std::vector<cv::Point> &pts);
	bool fit(const std::vector<cv::Point> &pts);
	bool fitRANSACSpline(cv::Size imgSize, const std::vector<cv::Point> &pts);
	
	// utility functions
	static void computeAccumulativeLength(const std::vector<cv::Point> &pts,
		std::vector<double> &sacc);
	static void computeAccumulativeLength(const cv::Mat &pts,
		std::vector<double> &sacc);
	static void constructSplineParameterMatrix(
		const std::vector<cv::Point> &pts, cv::Mat &Tmat);
	static void constructSplineParameterMatrix(
		int npt, cv::Mat &Tmat);
	
	// properties
	cv::Mat Mmat; // 4x4 bezier matrix
	cv::Mat Pmat; // 4x2 control point matrix

};


class LaneDetector
{
public:
	LaneDetector();
	~LaneDetector();

	// ========== Initialization Fcns ==================
	void constructPerspectiveMapping(cv::Size imgSize,
		double x, double y, double z, 
		double pan, double tilt, double roll);
	
	// construct look up table
	// @imgSize: input image size
	// @width: output image width defined in LUTs
	// @mppx: meter per pixel in x direction
	// @mppy: meter per pixel in y direction
	void constructLUT(cv::Size imgSize,
		int width, double mppx, double mppy);

	void initKF(cv::Size imgSize);

	// define ROI boundary
	// the ROI is in trapezoidal shape. 
	//
	//             xMin    xMax
	// top(nyMin)   -------
	// bot(nyMax  -----------
	// @nyMin: normalized y min [0,1]
	// @nyMax: normalized y max [0,1]
	// @nxMin_top: normalized x min at top
	// @nxMax_top: normalized x max at top
	// @nxMin_bot: normalized x min at bot
	// @nxMax_bot: normalized x max at bot
	void defineROI(double nyMin, double nyMax,
		double nxMin_top, double nxMax_top, 
		double nxMin_bot, double nxMax_bot);
	
	// ========== Pipeline Fcns ==========================
	
	// crop to region of interest 
	// used for the subsequent processing
	// @gray: only take gray image
	// @dst:: output image
	void cropToROI(const cv::Mat &gray, cv::Mat &dst);

	// parallel project camera image to ground image
	// @src: camera image
	// @imgG: output ground image
	void getGroundImage(const cv::Mat &src, cv::Mat &imgG);

	// obtain line by filtering
	// @grayG: gray-scale ground image
	// @edgeG: mask-like gray image showing lines (after filtering) 
	void getFilteredLines(const cv::Mat &grayG, cv::Mat &lineG);

	// grouping points
	// @gray: only take gray image
	// @lpts: output points on the left side 
	// @rpts: output points on the right side 
	bool groupPoints(const cv::Mat &gray,
		std::vector<cv::Point> &lpts, std::vector<cv::Point> &rpts);

	// get vertically scanned points closest to the centerline
	// for either right or left side 
	// @gray: input image in gray
	// @nDivX_2: number of division in x for half image (left and right)
	// @nDivY: number of division in y direction
	// @pts: output scanned points for larger to smaller y
	// @left: left(true) or right flag
	void getVerticalScannedPoints(const cv::Mat &gray,
		int nDivX_2, int nDivY, 
		std::vector<cv::Point> &pts, bool left);

	void findLaneByKFOld(const cv::Mat &gray,
		std::vector<cv::Point> &lanePts, bool left);

	void findLaneByKF(const cv::Mat &gray,
		std::vector<cv::Point> &lanePts, bool left);

	void getCamPtsFromGndImgPts(const std::vector<cv::Point> &gndImgPts,
		std::vector<cv::Point> &camPts);

	void detectLane(const cv::Mat &src, cv::Mat &gndView,
		cv::Mat &gndMarker, cv::Mat &gndGray, cv::Mat &bndGray,
		cv::Mat &dst);

	// ============ Utility Fcns =======================
	
	// mask the lane color, return the pixel in the 
	// color (white or yellow) range
	static void colorThresholding(const cv::Mat &src, cv::Mat &maskOut);
	
	// fit line
	// @pts: points 2i
	// @line: outcome line equation vector 3d (a,b,c) => 
	// ax+by+c=0
	static bool fitLine(const std::vector<cv::Point> pts, 
		cv::Vec3d &line);

	static void getPointsFromImage(const cv::Mat &gray,
		int uStart, int uEnd, int vStart, int vEnd,
		cv::Mat &points);


	// ============ Properties =================================
	cv::Mat mTM_vp; // transformation from vehicle to pixel frame
	
	// camera parameters
	double mFocalX = 0.028; 
	double mFocalY = 0.028;  
	double mCCDX = 0.01586;
	double mCCDY = 0.0132;

	// ROI constants
	double mROI_yTop= 0.55;
	double mROI_yBot= 0.9;
	double mROI_xminTop= 0.45;
	double mROI_xmaxTop= 0.55;
	double mROI_xminBot= 0.05;
	double mROI_xmaxBot= 0.95;

	// LUTs
	double mMPPx;
	double mMPPy;
	cv::Mat mXMap, mYMap, mZMap; // double format
	cv::Mat mLUT_u, mLUT_v;  // uchar format

	// kalman filters
	cv::KalmanFilter *mKFL;
	cv::KalmanFilter *mKFR;
};







