#pragma once

#include <functional>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2\highgui.hpp>
#include <iostream>

// ----- rotation matrix -----
void rotx(double ang, cv::Mat &rm);
void roty(double ang, cv::Mat &rm);
void rotz(double ang, cv::Mat &rm);
void rotzyx(double angz, double angy, double angx, cv::Mat &rm);
void rotyxz(double angy, double angx, double angz, cv::Mat &rm);

// normalize vector
//@ vec: cv::Mat column format
//@ homogeneous: whether the vector is a homogeneous vector
void normalizeVec(cv::Mat &vec, bool homogeneous);

// orthogonalize rotation matrix
void orthogonalizeRMatrix(cv::InputArray Rin, cv::OutputArray Rout);


// ----- camera intrinsic and extrinsic matrices -----
void getCameraIntrinsicMatrix3x3(double focalLengthX, double focalLengthY,
	double ccdWidth,double ccdHeight,
	double resolutionX, double resolutionY,
	cv::Mat &km);
void getCameraIntrinsicMatrix4x4(double focalLengthX, double focalLengthY,
	double ccdWidth,double ccdHeight, 
	double resolutionX, double resolutionY,
	cv::Mat &km);
void getCameraExtrinsicMatrix3x4(double x, double y, double z,
	double pan, double tilt, double roll,
	cv::Mat &rt);
void getCameraExtrinsicMatrix4x4(double x, double y, double z,
	double pan, double tilt, double roll,
	cv::Mat &rt);

// ----- detect calibration spheres -----
// @img: input image
// @centers: centers keypoint of the spheres
void detectCalibrationSpheres(cv::Mat img,
	std::vector<cv::Point2d> &centers);

// ----- sort functions ----- 
enum class SortSeq { AscThenDesc, AscThenAsc, DescThenAsc, DescThenDesc };
void sortXThenY(std::vector<cv::Point2d> &pts, SortSeq seq = SortSeq::AscThenAsc, double tol = 1e-3);
void sortYThenX(std::vector<cv::Point2d> &pts, SortSeq seq = SortSeq::AscThenAsc, double tol = 1e-3);

// ----- utility functions -----
double fsolve(std::function<double(double)> fx,
	std::function<double(double)> dfdx, double x0, bool printYes = false,
	double tol = 1e-3, int maxIter = 500);

// ----- rays -----
struct Ray3D
{
	cv::Point3d R0; // initial point
	cv::Point3d Rd; // normalized direction
	void getPoint(double t, cv::Point3d &pt) const
	{
		pt= R0 + (Rd*t);
	};
	void setDirection(double x, double y, double z)
	{
		double mag = std::sqrt(x*x + y*y + z*z);
		if (mag == 0)
		{
			throw std::overflow_error("ray direction magnitude is zero");
		}
		else
		{
			Rd.x = x / mag;
			Rd.y = y / mag;
			Rd.z = z / mag;
		}
	};
	void printInfo();
};

struct Plane3D
{
	cv::Point3d n; // normal vector (normalized)
	double D;  // offset value

	void setNormal(double x, double y, double z)
	{
		double mag = std::sqrt(x*x + y*y + z*z);
		if (mag == 0)
		{
			throw std::overflow_error("normal vector has zero length!");
		}
		else
		{
			n.x = x / mag; n.y = y / mag; n.z = z / mag;
		}
	};

	// intersect with a ray
	// @ray: ray to be intersected
	// @hitPt: output hit point, if any
	// @tMin: minimum parameter t for the intersection to be valid
	// @return true if there is a valid intersection
	bool intersect(const Ray3D &ray, cv::Point3d &hitPt, double tMin) const;
};

struct Cylinder
{
	cv::Point3d n; // axial normal vector
	cv::Point3d R0; // origin(base) point
	double r; // radius
	
	void setNormal(double x, double y, double z)
	{
		double mag = std::sqrt(x*x + y*y + z*z);
		if (mag == 0)
		{
			throw std::overflow_error("normal vector has zero length!");
		}
		else
		{
			n.x = x / mag; n.y = y / mag; n.z = z / mag;
		}
	};
	// intersect with a ray
	// @ray: ray to be intersected
	// @hitPt: output hit point, if any
	// @tMin: minimum parameter t for the intersection to be valid
	// @return true if there is a valid intersection
	bool intersect(const Ray3D &ray, cv::Point3d &hitPt, double tMin) const;
};

struct HybridShape
{
	HybridShape(double radius = 30, 
		double paraCoef= 30):
		mC(radius), mA(paraCoef)
	{ }

	double mC; // paraboloid "radius" 
	double mA; // paraboloid coeff.

	// intersect with a ray
	// @ray: ray to be intersected
	// @hitPt: output hit point, if any
	// @tMin: minimum parameter t for the intersection to be valid
	// @return true if there is a valid intersection
	bool intersect(const Ray3D &ray, cv::Point3d &hitPt, double tMin) const;
};

struct GroundedCylinder
{
	GroundedCylinder(double radius = 30) :
		mRadius(radius)
	{
		mGroundPlane.D = 0;
		mGroundPlane.setNormal(0, 0, 1);
		mWall.R0 = cv::Point3d(0, 0, 0);
		mWall.r = mRadius;
		mWall.setNormal(0, 0, 1);
	}	
	
	double mRadius;
	Plane3D mGroundPlane;
	Cylinder mWall;
	
	// intersect with a ray
	// @ray: ray to be intersected
	// @hitPt: output hit point, if any
	// @tMin: minimum parameter t for the intersection to be valid
	// @return true if there is a valid intersection
	bool intersect(const Ray3D &ray, cv::Point3d &hitPt, double tMin) const;
};

enum class ProjectionModelType { GroundPlane, Hybrid, Cylinder };

class VirtualCamera
{
public:
	VirtualCamera() {};
	VirtualCamera(double x_v,double y_v, double z_v,
		double pan, double tilt, double roll);
	~VirtualCamera() {};

	void setPose(double x_v, double y_v, double z_v,
		double pan, double tilt, double roll);

	void generateRays(std::vector<Ray3D> &rays);
	void hitPlane(const Plane3D &plane,
		cv::Mat &xCoord, cv::Mat &yCoord, cv::Mat &zCoord);
	void hitHybrid(const HybridShape &hs,
		cv::Mat &xCoord, cv::Mat &yCoord, cv::Mat &zCoord);
	void hitCylinder(const GroundedCylinder &gc,
		cv::Mat &xCoord, cv::Mat &yCoord, cv::Mat &zCoord);
		
	double fovAngle = 100 * CV_PI / 180;
	double aspect = 16.0 / 9.0;
	int resolutionX = 800;
	// camera pose in vehicle coordinates
	double mX_v = 0;
	double mY_v = 0;
	double mZ_v = 10;
	double mPan = 0;
	double mTilt = 0;
	double mRoll = 0;

	// camera transformation matrix
	cv::Mat mTM;

	// virtual camera lookup table
	std::vector<cv::Mat> mLUT_CamNo;
	std::vector<cv::Mat> mLUT_u;
	std::vector<cv::Mat> mLUT_v;	
};
