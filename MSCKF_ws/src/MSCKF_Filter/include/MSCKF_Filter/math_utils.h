#ifndef _MATH_UTILS_H_
#define _MATH_UTILS_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cstdlib>
#include <math.h>



namespace math_utils {

struct gpsOdom{
    float x;
    float y;
    float z;
    float yaw;
};    
    
    
template<typename Derived>   
static int sign(Derived x) {
    if (x >= static_cast<Derived>(0))
	return 1;
    else
	return -1;
} 
    
template<typename Type>
static Type wrap_pi(Type x) {
    while (x >= Type(M_PI)) {
	x -= Type(2.0 * M_PI);
    }

    while (x < Type(-M_PI)) {
	x += Type(2.0 * M_PI);
    }
    return x;
}
    
static void enforceSymmetry(Eigen::MatrixXd& mat){
  mat = 0.5*(mat+mat.transpose()).eval();
}

// static void enforceNormal(Eigen::Quaterniond& quat){
//     double eq = 0.5*(quat.norm() - 1);
//     quat = (1 - eq)*quat;
// }



    
    
static Eigen::Quaterniond axis2Quat(const Eigen::Vector3d &axis, double theta) {
    Eigen::Quaterniond q;

    if (theta < 1e-10) {
	q.w() = 1.0;
	q.x() = q.y() = q.z() = 0;
    }

    double magnitude = sin(theta / 2.0f);

    q.w() = cos(theta / 2.0f);
    q.x() = axis(0) * magnitude;
    q.y() = axis(1) * magnitude;
    q.z() = axis(2) * magnitude;
    
    return q;
}

static Eigen::Quaterniond axis2Quat(const Eigen::Vector3d &vec) {
    Eigen::Quaterniond q;
    double theta = vec.norm();

    if (theta < 1e-10) {
	q.w() = 1.0;
	q.x() = q.y() = q.z() = 0;
	return q;
    }

    Eigen::Vector3d tmp = vec / theta;
    return axis2Quat(tmp, theta);
}
    
static Eigen::Vector3d Quat2axis(const Eigen::Quaterniond &q) {
    double axis_magnitude = sqrt(q.x() * q.x() + q.y() * q.y() + q.z() * q.z());
    Eigen::Vector3d vec;
    vec(0) = q.x();
    vec(1) = q.y();
    vec(2) = q.z();

    if (axis_magnitude >= 1e-10) {
	vec = vec / axis_magnitude;
	vec = vec * wrap_pi(2.0 * atan2(axis_magnitude, q.w()));
    }

    return vec;
} 
    
    
    
template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(const Eigen::MatrixBase<Derived> &ypr)
{
    typedef typename Derived::Scalar Scalar_t;

    Scalar_t y = ypr(0);
    Scalar_t p = ypr(1);
    Scalar_t r = ypr(2);

    Eigen::Matrix<Scalar_t, 3, 3> Rz;
    Rz << cos(y), -sin(y), 0,
	sin(y), cos(y), 0,
	0, 0, 1;

    Eigen::Matrix<Scalar_t, 3, 3> Ry;
    Ry << cos(p), 0., sin(p),
	0., 1., 0.,
	-sin(p), 0., cos(p);

    Eigen::Matrix<Scalar_t, 3, 3> Rx;
    Rx << 1., 0., 0.,
	0., cos(r), -sin(r),
	0., sin(r), cos(r);

    return Rz * Ry * Rx;
}



//void setRPY(const Derived& roll, const Derived& pitch, const Derived& yaw)
template <typename Derived>
static Eigen::Quaternion<Derived> ypr2Quat(const Eigen::MatrixBase<Derived> &ypr)
{
    Derived halfYaw = Derived(ypr(0)) * Derived(0.5);  
    Derived halfPitch = Derived(ypr(1)) * Derived(0.5);  
    Derived halfRoll = Derived(ypr(2)) * Derived(0.5);  
    Derived cosYaw = cos(halfYaw);
    Derived sinYaw = sin(halfYaw);
    Derived cosPitch = cos(halfPitch);
    Derived sinPitch = sin(halfPitch);
    Derived cosRoll = cos(halfRoll);
    Derived sinRoll = sin(halfRoll);
    Eigen::Quaternion<Derived> Q;
    Q.x() = sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw;
    Q.y() = cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw;
    Q.z() = cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw;
    Q.w() = cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw;
    Q.normalized();
    return Q;
}

static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R) {
    Eigen::Vector3d ypr;
    ypr(1) = -asin(R(2,0)); //pitch
    ypr(2) = atan2(R(2,1) / cos(ypr(1)), R(2,2) / cos(ypr(1))); //roll
    ypr(0) = atan2(R(1,0) / cos(ypr(1)), R(0,0) / cos(ypr(1))); //yaw
    return ypr;
}

static Eigen::Vector3d Quat2ypr(const Eigen::Quaterniond &Q) {
    return R2ypr(Q.toRotationMatrix());
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3> rpy2R(const Eigen::MatrixBase<Derived> &rpy)
{
    typedef typename Derived::Scalar Scalar_t;

    Scalar_t r = rpy(0);
    Scalar_t p = rpy(1);
    Scalar_t y = rpy(2);
    
    Eigen::Matrix<Scalar_t, 3, 3> Rz;
    Rz << cos(y), -sin(y), 0,
	sin(y), cos(y), 0,
	0, 0, 1;

    Eigen::Matrix<Scalar_t, 3, 3> Ry;
    Ry << cos(p), 0., sin(p),
	0., 1., 0.,
	-sin(p), 0., cos(p);

    Eigen::Matrix<Scalar_t, 3, 3> Rx;
    Rx << 1., 0., 0.,
	0., cos(r), -sin(r),
	0., sin(r), cos(r);

    return Rz * Ry * Rx;
}

template <typename Derived>
static Eigen::Quaternion<typename Derived::Scalar> rpy2Quat(const Eigen::MatrixBase<Derived> &rpy)
{
    Eigen::Matrix<typename Derived::Scalar, 3, 3> R = rpy2R(rpy);
    Eigen::Quaternion<typename Derived::Scalar> Q(R);
    return Q;
}

static Eigen::Vector3d R2rpy(const Eigen::Matrix3d &R) {
    Eigen::Vector3d rpy;
    rpy(1) = -asin(R(2,0)); //pitch
    rpy(0) = atan2(R(2,1) / cos(rpy(1)), R(2,2) / cos(rpy(1))); //roll
    rpy(2) = atan2(R(1,0) / cos(rpy(1)), R(0,0) / cos(rpy(1))); //yaw
    return rpy;
}

static Eigen::Vector3d Q2rpy(const Eigen::Quaterniond &Q) {
    return R2rpy(Q.toRotationMatrix());
}



template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3> skew(const Eigen::MatrixBase<Derived> &q)
{
    Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
    ans << typename Derived::Scalar(0), -q(2), q(1),
	q(2), typename Derived::Scalar(0), -q(0),
	-q(1), q(0), typename Derived::Scalar(0);
    return ans;
}   


template <typename Derived>
static Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta)
{
    typedef typename Derived::Scalar Scalar_t;

    Eigen::Quaternion<Scalar_t> dq;
    Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
    half_theta /= static_cast<Scalar_t>(2.0);
    dq.w() = static_cast<Scalar_t>(1.0);
    dq.x() = half_theta.x();
    dq.y() = half_theta.y();
    dq.z() = half_theta.z();
    return dq;
}

template <class T>
static T rad2deg(const T& radians)
{
    return radians * 180.0 / M_PI;
}

template <class T>
static T deg2rad(const T& degrees)
{
    return degrees * M_PI / 180.0;
} 

template <class Derived>
static Eigen::MatrixBase<Derived> rad2deg(const Eigen::MatrixBase<Derived> radians)
{
    return Eigen::MatrixBase<Derived>(rad2deg(radians(0)), rad2deg(radians(1)), rad2deg(radians(2)));
}

template <class Derived>
static Eigen::MatrixBase<Derived> deg2rag(const Eigen::MatrixBase<Derived> degrees)
{
    return Eigen::MatrixBase<Derived>(deg2rad(degrees(0)), deg2rad(degrees(1)), deg2rad(degrees(2)));
}

}





#endif 