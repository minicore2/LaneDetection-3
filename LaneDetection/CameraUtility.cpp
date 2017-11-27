#include "CameraUtility.h"

void rotx(double ang, cv::Mat &rm)
{
	rm = cv::Mat::zeros(3, 3, CV_64FC1);

	double cth = std::cos(ang);
	double sth = std::sin(ang);

	rm.at<double>(0, 0) = 1.0;
	rm.at<double>(1, 1) = cth;
	rm.at<double>(1, 2) = -sth;
	rm.at<double>(2, 1) = sth;
	rm.at<double>(2, 2) = cth;
}

void roty(double ang, cv::Mat &rm)
{
	rm = cv::Mat::zeros(3, 3, CV_64FC1);

	double cth = std::cos(ang);
	double sth = std::sin(ang);

	rm.at<double>(1, 1) = 1.0;
	rm.at<double>(0, 0) = cth;
	rm.at<double>(0, 2) = sth;
	rm.at<double>(2, 0) = -sth;
	rm.at<double>(2, 2) = cth;
}

void rotz(double ang, cv::Mat &rm)
{
	rm = cv::Mat::zeros(3, 3, CV_64FC1);

	double cth = std::cos(ang);
	double sth = std::sin(ang);

	rm.at<double>(2, 2) = 1.0;
	rm.at<double>(0, 0) = cth;
	rm.at<double>(0, 1) = -sth;
	rm.at<double>(1, 0) = sth;
	rm.at<double>(1, 1) = cth;
}
// rotation matrix using post-multiplication (local axis)
void rotzyx(double angz, double angy, double angx, cv::Mat &rm)
{
	rm = cv::Mat::eye(3, 3, CV_64FC1);
	cv::Mat tmp(3, 3, CV_64FC1);
	rotz(angz, tmp);
	rm = rm* tmp;
	roty(angy, tmp);
	rm = rm* tmp;
	rotx(angx, tmp);
	rm = rm* tmp;
}

void rotyxz(double angy, double angx, double angz, cv::Mat & rm)
{
	rm = cv::Mat::eye(3, 3, CV_64FC1);
	cv::Mat tmp(3, 3, CV_64FC1);
	roty(angy, tmp);
	rm = rm* tmp;
	rotx(angx, tmp);
	rm = rm* tmp;
	rotz(angz, tmp);
	rm = rm* tmp;
}


void normalizeVec(cv::Mat & vec, bool homogeneous)
{
	double x = 0;
	int nrow = vec.rows;
	if (homogeneous)
	{
		nrow = vec.rows - 1;
	}
	for (int i = 0; i < nrow; ++i)
	{
		x += vec.at<double>(i, 0)*vec.at<double>(i, 0);
	}
	x = std::sqrt(x);
	for (int i = 0; i < nrow; ++i)
	{
		vec.at<double>(i, 0) /= x;
	}
}
void orthogonalizeRMatrix(cv::InputArray Rin, cv::OutputArray Rout)
{
	cv::Mat U, D, Vt, out;
	cv::SVDecomp(Rin, D, U, Vt);
	Rout.create(Rin.size(), Rin.type());		
	out = U*Vt;
	out.copyTo(Rout);
}


void getCameraIntrinsicMatrix3x3(double focalLengthX, double focalLengthY,
	double ccdWidth, double ccdHeight,
	double resolutionX, double resolutionY,
	cv::Mat &km)
{
	km = cv::Mat::zeros(3, 3, CV_64FC1);
	double scaleFactorX = ccdWidth / resolutionX;
	double scaleFactorY = ccdHeight / resolutionY;

	km.at<double>(0, 0) = focalLengthX / scaleFactorX;
	km.at<double>(1, 1) = focalLengthY / scaleFactorY;
	km.at<double>(2, 2) = 1.0;
	km.at<double>(0, 2) = resolutionX / 2;
	km.at<double>(1, 2) = resolutionY / 2;
}

void getCameraIntrinsicMatrix4x4(double focalLengthX, double focalLengthY,
	double ccdWidth, double ccdHeight,
	double resolutionX, double resolutionY,
	cv::Mat &km)
{
	km = cv::Mat::eye(4, 4, CV_64FC1);
	double scaleFactorX = ccdWidth / resolutionX;
	double scaleFactorY = ccdHeight / resolutionY;

	km.at<double>(0, 0) = focalLengthX / scaleFactorX;
	km.at<double>(1, 1) = focalLengthY / scaleFactorY;
	km.at<double>(2, 2) = 1.0;
	km.at<double>(0, 2) = resolutionX / 2;
	km.at<double>(1, 2) = resolutionY / 2;
}

void getCameraExtrinsicMatrix3x4(double x, double y, double z,
	double pan, double tilt, double roll, cv::Mat & rt)
{
	rt = cv::Mat::zeros(3, 4, CV_64FC1);
	cv::Mat rm_cv, tmp;
	// rotate ANVEL axis to align with the camera plane 
	rotzyx(-0.5*CV_PI,0, -0.5*CV_PI, rm_cv);
	
	// rotate camera along its local axis now
	rotyxz(pan, tilt, roll, tmp);
	rm_cv = rm_cv*tmp;
	
	// fill in rotation
	cv::Mat rm_vc = rm_cv.t();
	cv::Rect roi(0, 0, 3, 3);
	rm_vc.copyTo(rt(roi));

	// translation
	cv::Mat t_cv(3, 1, CV_64FC1), t_vc;
	t_cv.at<double>(0, 0) = x;
	t_cv.at<double>(1, 0) = y;
	t_cv.at<double>(2, 0) = z;
	t_vc = -rm_vc*t_cv;

	rt.at<double>(0, 3) = t_vc.at<double>(0, 0);
	rt.at<double>(1, 3) = t_vc.at<double>(1, 0);
	rt.at<double>(2, 3) = t_vc.at<double>(2, 0);
}

void getCameraExtrinsicMatrix4x4(double x, double y, double z, 
	double pan, double tilt, double roll, cv::Mat & rt)
{
	rt = cv::Mat::eye(4, 4, CV_64FC1);

	cv::Mat rm_cv, tmp;
	// rotate ANVEL axis to align with the camera plane 
	rotzyx(-0.5*CV_PI,0, -0.5*CV_PI, rm_cv);
	
	// rotate camera along its local axis now
	rotyxz(pan, tilt, roll, tmp);
	rm_cv = rm_cv*tmp;

	// fill in rotation
	cv::Mat rm_vc = rm_cv.t();
	cv::Rect roi(0, 0, 3, 3);
	rm_vc.copyTo(rt(roi));

	// translation
	cv::Mat t_cv(3, 1, CV_64FC1), t_vc;
	t_cv.at<double>(0, 0) = x;
	t_cv.at<double>(1, 0) = y;
	t_cv.at<double>(2, 0) = z;
	t_vc = -rm_vc*t_cv;

	rt.at<double>(0, 3) = t_vc.at<double>(0, 0);
	rt.at<double>(1, 3) = t_vc.at<double>(1, 0);
	rt.at<double>(2, 3) = t_vc.at<double>(2, 0);
}


void detectCalibrationSpheres(cv::Mat img,
	std::vector<cv::Point2d> &centers)
{
	// convert to hsv
	cv::Mat hsv, maskR, maskC, maskO, mask;
	cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

	// get mask in color range (red, cyan , orange) 
	cv::inRange(hsv, cv::Scalar(0, 220, 50),
		cv::Scalar(30, 255, 255), maskR);
	cv::inRange(hsv, cv::Scalar(90, 220, 50),
		cv::Scalar(150, 255, 255), maskC);
	cv::inRange(hsv, cv::Scalar(0, 220, 50),
		cv::Scalar(30, 255, 255), maskO);

	// create combined mask
	cv::bitwise_or(maskR, maskC, mask);
	cv::bitwise_or(maskO, mask, mask);

	cv::Mat maskInv;
	cv::bitwise_not(mask, maskInv);

	// get only spheres 
	cv::Mat dst;
	cv::bitwise_and(img, img, dst, mask);

	// change background to white	
	for (int i = 0; i < dst.rows; ++i)
	{
		for (int j = 0; j < dst.cols; ++j)
		{
			if (mask.at<uchar>(i, j) < 100)
			{
				dst.at<cv::Vec4b>(i, j) = cv::Vec4b(255, 255, 255, 255);
			}
		}
	}

	// find circle grid
	cv::SimpleBlobDetector::Params params;
	params.minThreshold = 200;
	params.maxThreshold = 255;
	params.filterByCircularity = false;
	params.filterByArea = true;
	params.minArea = 5;
	params.filterByConvexity = false;

	cv::Ptr<cv::SimpleBlobDetector> bd =
		cv::SimpleBlobDetector::create(params);
	//cv::Mat gray;
	//cv::cvtColor(dst, gray, cv::COLOR_BGR2GRAY);
	
	cv::imshow("testa", img);
	cv::waitKey(0);

	std::vector<cv::KeyPoint> kps;
	bd->detect(maskInv, kps);

	// store to centers
	centers.clear();
	for (cv::KeyPoint kp : kps)
	{
		centers.push_back(kp.pt);
	}
}

void sortXThenY(std::vector<cv::Point2d>& pts, SortSeq seq,
	double tol)
{
	// sort by x then by y
	if (seq == SortSeq::AscThenAsc)
	{
		std::sort(pts.begin(), pts.end(),
			[tol](auto pt1, auto pt2) {return
			((pt1.x < (pt2.x - tol)) ||
			(std::abs(pt1.x - pt2.x) <= tol && (pt1.y < (pt2.y - tol)))); });
	}
	else if (seq == SortSeq::AscThenDesc)
	{
		std::sort(pts.begin(), pts.end(),
			[tol](auto pt1, auto pt2) {return
			((pt1.x < (pt2.x - tol)) ||
			(std::abs(pt1.x - pt2.x) <= tol && (pt1.y >(pt2.y + tol)))); });
	}
	else if (seq == SortSeq::DescThenAsc)
	{
		std::sort(pts.begin(), pts.end(),
			[tol](auto pt1, auto pt2) {return
			((pt1.x > (pt2.x + tol)) ||
			(std::abs(pt1.x - pt2.x) <= tol && (pt1.y < (pt2.y - tol)))); });
	}
	else //SortSeq::DescThenDesc
	{
		std::sort(pts.begin(), pts.end(),
			[tol](auto pt1, auto pt2) {return
			((pt1.x >(pt2.x + tol)) ||
			(std::abs(pt1.x - pt2.x) <= tol && (pt1.y > (pt2.y + tol)))); });
	}

}

void sortYThenX(std::vector<cv::Point2d>& pts, SortSeq seq,
	double tol)
{
	// sort by y then by x
	if (seq == SortSeq::AscThenAsc)
	{
		std::sort(pts.begin(), pts.end(),
			[tol](auto pt1, auto pt2) {return
			((pt1.y < (pt2.y - tol)) ||
			(std::abs(pt1.y - pt2.y) <= tol && (pt1.x < (pt2.x - tol)))); });
	}
	else if (seq == SortSeq::AscThenDesc)
	{
		std::sort(pts.begin(), pts.end(),
			[tol](auto pt1, auto pt2) {return
			((pt1.y < (pt2.y - tol)) ||
			(std::abs(pt1.y - pt2.y) <= tol && (pt1.x >(pt2.x + tol)))); });
	}
	else if (seq == SortSeq::DescThenAsc)
	{
		std::sort(pts.begin(), pts.end(),
			[tol](auto pt1, auto pt2) {return
			((pt1.y > (pt2.y + tol)) ||
			(std::abs(pt1.y - pt2.y) <= tol && (pt1.x < (pt2.x - tol)))); });
	}
	else //SortSeq::DescThenDesc
	{
		std::sort(pts.begin(), pts.end(),
			[tol](auto pt1, auto pt2) {return
			((pt1.y >(pt2.y + tol)) ||
			(std::abs(pt1.y - pt2.y) <= tol && (pt1.x > (pt2.x + tol)))); });
	}
}

double fsolve(std::function<double(double)> fx, 
	std::function<double(double)> dfdx, double x0, bool printYes,
	double tol, int maxIter)
{
	double xn = x0;
	double residual = std::abs(fx(xn));
	double dx;
	int iter = 0;
	
	while (residual > tol && iter < maxIter)
	{
		dx = fx(xn) / std::max(dfdx(xn), 1e-10);
		xn = xn - dx;
		residual = std::abs(fx(xn));
		if (printYes)
		{
			std::cout << "iter " << iter << ": x= " << xn << " f(x)= " << residual << "\n";
		}
		++iter;		
	}
	return xn;
}

bool Plane3D::intersect(const Ray3D & ray, 
	cv::Point3d & hitPt, double tMin) const
{
	double num, den, t;
	num = -(D + n.dot(ray.R0));
	den = n.dot(ray.Rd);

	bool intersectYes= false;
	if (den != 0)
	{
		t = num / den;
		if (t >= tMin)
		{
			intersectYes = true;
			ray.getPoint(t, hitPt); // get the point
		}		
	}	

	return intersectYes;
}

bool Cylinder::intersect(const Ray3D & ray, 
	cv::Point3d & hitPt, double tMin) const
{
	// ---get coefficient of the equation---
	// first term
	cv::Point3d Avec;
	double A;
	Avec = ray.Rd - (ray.Rd.dot(n))*n;
	A = Avec.dot(Avec);

	bool intersectYes = false;
	if (A == 0)  // ray and cylinder parallel, no intersection
	{
		return intersectYes;
	}

	// second term
	cv::Point3d Bvec1, Bvec2, dp;
	double B;
	dp = ray.R0 - R0;
	Bvec1 = ray.Rd - (ray.Rd.dot(n))*n;
	Bvec2 = dp - (dp.dot(n))*n;
	B = 2 * (Bvec1.dot(Bvec2));

	// third term
	cv::Point3d Cvec;
	double C;
	Cvec = dp - (dp.dot(n))*n;
	C = (Cvec.dot(Cvec)) - r*r;

	// solve for t
	double t, t1,t2;
	
	double termS = B*B - 4 * A*C;
	
	if (termS ==0)
	{
		t = -B / 2 / A;
		if (t >= tMin)
		{
			intersectYes = true;
			ray.getPoint(t, hitPt); // get the point
		}
	}
	else if (termS > 0)
	{
		// since A>0, => t1 > t2
		t1 = -B / 2 / A + std::sqrt(termS) / 2 / A;
		t2 = -B / 2 / A - std::sqrt(termS) / 2 / A;
		if (t2 >= tMin)
		{
			intersectYes = true;
			ray.getPoint(t2, hitPt); // get the point
		}
		else if (t1 >= tMin)
		{
			intersectYes = true;
			ray.getPoint(t1, hitPt); // get the point
		}
	}

	return intersectYes;
}

VirtualCamera::VirtualCamera(double x_v, double y_v, double z_v,
	double pan, double tilt, double roll) :
	mX_v(x_v), mY_v(y_v), mZ_v(z_v),
	mPan(pan), mTilt(tilt), mRoll(roll)
{
	setPose(x_v, y_v, z_v, pan, tilt, roll);
}

void VirtualCamera::generateRays(std::vector<Ray3D> &rays)
{
	// convert pixels location to camera coordinates
	int resolutionY = static_cast<int>(resolutionX / aspect);

	// calculate parameters
	double D = 1 / std::tan(fovAngle / 2);

	// get rotation matrix
	cv::Mat rm;
	cv::Rect roi(0, 0, 3, 3);
	mTM(roi).copyTo(rm);
	
	// create each ray
	rays.clear();
	Ray3D ray; ray.R0 = cv::Point3d(mX_v, mY_v, mZ_v);
	double xc, yc;
	cv::Vec3d vDir; vDir(2) = D;
	cv::Mat result;
	for (int i = 0; i < resolutionX; ++i)
	{
		xc = 2*i / double(resolutionX) - 1.0;
		for (int j = 0; j < resolutionY; ++j)
		{
			yc = 2 * j / double(resolutionY) - 1.0;
			vDir(0) = xc;
			vDir(1) = yc;
			result = rm*cv::Mat(vDir);
			ray.setDirection(result.at<double>(0,0),
					result.at<double>(1,0),
					result.at<double>(2,0));
			rays.push_back(ray);
		}
	}
}

void VirtualCamera::hitPlane(const Plane3D & plane, cv::Mat & xCoord, cv::Mat & yCoord, cv::Mat & zCoord)
{
	// initialize matrix
	int resolutionY = static_cast<int>(resolutionX / aspect);
	xCoord.create(resolutionY, resolutionX, CV_64FC1);
	yCoord.create(resolutionY, resolutionX, CV_64FC1);
	zCoord.create(resolutionY, resolutionX, CV_64FC1);

	// calculate parameters
	double D = 1 / std::tan(fovAngle / 2);

	// get rotation matrix
	cv::Mat rm, rm_t;
	cv::Rect roi(0, 0, 3, 3);
	mTM(roi).copyTo(rm);
	rm_t = rm.t();

	// create each ray
	Ray3D ray; ray.R0 = cv::Point3d(mX_v, mY_v, mZ_v);
	double xc, yc;  // camera coordinate
	cv::Vec3d vDir; vDir(2) = D;
	cv::Mat result;
	cv::Point3d hitPt;
	for (int i = 0; i < resolutionX; ++i)
	{
		xc = 2 * i / double(resolutionX) - 1.0;
		for (int j = 0; j < resolutionY; ++j)
		{
			// set the ray			
			yc = 2 * j / double(resolutionY) - 1.0;
			vDir(0) = xc;
			vDir(1) = yc;
			result = rm_t*cv::Mat(vDir); // convert to global
			ray.setDirection(result.at<double>(0, 0),
				result.at<double>(1, 0),
				result.at<double>(2, 0));
			// get intersection
			if (plane.intersect(ray, hitPt, 0))
			{
				xCoord.at<double>(j, i) = hitPt.x;
				yCoord.at<double>(j, i) = hitPt.y;
				zCoord.at<double>(j, i) = hitPt.z;
			}
		}
	}

}

void VirtualCamera::hitHybrid(const HybridShape & hs, cv::Mat & xCoord, cv::Mat & yCoord, cv::Mat & zCoord)
{
	// initialize matrix
	int resolutionY = static_cast<int>(resolutionX / aspect);
	xCoord.create(resolutionY, resolutionX, CV_64FC1);
	yCoord.create(resolutionY, resolutionX, CV_64FC1);
	zCoord.create(resolutionY, resolutionX, CV_64FC1);

	// calculate parameters
	double D = 1 / std::tan(fovAngle / 2);

	// get rotation matrix
	cv::Mat rm, rm_t;
	cv::Rect roi(0, 0, 3, 3);
	mTM(roi).copyTo(rm);
	rm_t = rm.t();

	// create each ray
	Ray3D ray; ray.R0 = cv::Point3d(mX_v, mY_v, mZ_v);
	double xc, yc;  // camera coordinate
	cv::Vec3d vDir; vDir(2) = D;
	cv::Mat result;
	cv::Point3d hitPt;
	for (int i = 0; i < resolutionX; ++i)
	{
		xc = 2 * i / double(resolutionX) - 1.0;
		for (int j = 0; j < resolutionY; ++j)
		{
			// set the ray			
			yc = 2 * j / double(resolutionY) - 1.0;
			vDir(0) = xc;
			vDir(1) = yc;
			result = rm_t*cv::Mat(vDir); // convert to global
			ray.setDirection(result.at<double>(0, 0),
				result.at<double>(1, 0),
				result.at<double>(2, 0));
			// get intersection
			if (hs.intersect(ray, hitPt, 0.1))
			{
				xCoord.at<double>(j, i) = hitPt.x;
				yCoord.at<double>(j, i) = hitPt.y;
				zCoord.at<double>(j, i) = hitPt.z;
			}
		}
	}
}

void VirtualCamera::hitCylinder(const GroundedCylinder & gc, cv::Mat & xCoord, cv::Mat & yCoord, cv::Mat & zCoord)
{
	// initialize matrix
	int resolutionY = static_cast<int>(resolutionX / aspect);
	xCoord.create(resolutionY, resolutionX, CV_64FC1);
	yCoord.create(resolutionY, resolutionX, CV_64FC1);
	zCoord.create(resolutionY, resolutionX, CV_64FC1);

	// calculate parameters
	double D = 1 / std::tan(fovAngle / 2);

	// get rotation matrix
	cv::Mat rm, rm_t;
	cv::Rect roi(0, 0, 3, 3);
	mTM(roi).copyTo(rm);
	rm_t = rm.t();

	// create each ray
	Ray3D ray; ray.R0 = cv::Point3d(mX_v, mY_v, mZ_v);
	double xc, yc;  // camera coordinate
	cv::Vec3d vDir; vDir(2) = D;
	cv::Mat result;
	cv::Point3d hitPt;
	for (int i = 0; i < resolutionX; ++i)
	{
		xc = 2 * i / double(resolutionX) - 1.0;
		for (int j = 0; j < resolutionY; ++j)
		{
			// set the ray			
			yc = 2 * j / double(resolutionY) - 1.0;
			vDir(0) = xc;
			vDir(1) = yc;
			result = rm_t*cv::Mat(vDir);  // convert to global
			ray.setDirection(result.at<double>(0, 0),
				result.at<double>(1, 0),
				result.at<double>(2, 0));
			// get intersection
			if (gc.intersect(ray, hitPt, 0.1))
			{
				xCoord.at<double>(j, i) = hitPt.x;
				yCoord.at<double>(j, i) = hitPt.y;
				zCoord.at<double>(j, i) = hitPt.z;
			}
		}
	}
}

void VirtualCamera::setPose(double x_v, double y_v, double z_v,
	double pan, double tilt, double roll)
{
	mX_v = x_v;
	mY_v = y_v;
	mZ_v = z_v;
	mPan = pan;
	mTilt = tilt;
	mRoll = roll;
	getCameraExtrinsicMatrix4x4(x_v, y_v, z_v, 
		mPan, mTilt, mRoll, mTM);
}

void Ray3D::printInfo()
{
	std::cout << "ray origin: " << R0 << " | direction: " << Rd << "\n";
}

bool HybridShape::intersect(const Ray3D & ray, 
	cv::Point3d & hitPt, double tMin) const
{
	bool intersectYes = false;
	// get intersection with the ground 
	cv::Point3d hitG;
	double c4, a4;
	c4 = std::pow(mC, 4);
	a4 = std::pow(mA, 4);

	// calculate ray direction magnitude on ground
	double magGround = std::sqrt(ray.Rd.x*ray.Rd.x + ray.Rd.y*ray.Rd.y);

	if (magGround != 0) // ray not parallel to the global z axis
	{
		// assign variables
		double r0x = ray.R0.x;
		double rdx = ray.Rd.x;
		double r0y = ray.R0.y;
		double rdy = ray.Rd.y;
		double r0z = ray.R0.z;
		double rdz = ray.Rd.z;

		// solve for intersection on bowl
		double tBowl =
			fsolve([&](double t) {return std::pow(r0x + rdx*t, 4)
				+ std::pow(r0y + rdy*t, 4)
				- a4*(r0z + rdz*t); },
				[&](double t) {return 4 * rdx*std::pow(r0x + rdx*t, 3)
				+ 4 * rdy*std::pow(r0y + rdy*t, 3)
				- rdz*a4; },
				tMin);
		double xBowl = r0x + rdx*tBowl;
		
		double yBowl = r0y + rdy*tBowl;
		double zBowl = r0z + rdz*tBowl;
		double xBowl4 = std::pow(xBowl, 4);
		double yBowl4 = std::pow(yBowl, 4);
				
		if (xBowl4+yBowl4<=c4) // inbound
		{
			intersectYes = true;
			hitPt.x = xBowl;
			hitPt.y = yBowl;
			hitPt.z = zBowl;			
		}
		else // out of bound
		{	
			intersectYes = true;
			double tb = fsolve([r0x, r0y, rdx, rdy, c4](double t)->double {
				return std::pow(r0x + rdx*t, 4) +
					std::pow(r0y + rdy*t, 4) - c4; },
				[r0x, r0y, rdx, rdy, c4](double t)->double {
					return 4 * rdx*std::pow(r0x + rdx*t, 3) +
						4 * rdy*std::pow(r0y + rdy*t, 3); },
					tMin);

			//std::cout << "tb= " << tb << " x= " << r0x + rdx*tb << " y= " << r0y + rdy*tb << "\n";
			// calculate 3D line t
			double ang = std::atan2(ray.Rd.z, std::sqrt(rdx*rdx + rdy*rdy));
			//double t = tb / std::cos(ang);

			hitPt.x = r0x + rdx*tb;
			hitPt.y = r0y + rdy*tb;
			hitPt.z = ray.R0.z + ray.Rd.z*tb;
		}
	}

	return intersectYes;
}

bool GroundedCylinder::intersect(const Ray3D & ray, cv::Point3d & hitPt, double tMin) const
{
	cv::Point3d hitW;
	bool intersectYes = false;
	if (mWall.intersect(ray, hitW, tMin))
	{
		intersectYes = true;
		if (hitW.z >= 0)  // hit the wall
		{
			hitPt.x = hitW.x;
			hitPt.y = hitW.y;
			hitPt.z = hitW.z;
		}
		else // hit the ground
		{
			if (!mGroundPlane.intersect(ray, hitPt, tMin))
			{
				intersectYes = false;
			}
		}
	}	
	return intersectYes;
}
