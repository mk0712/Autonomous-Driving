#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {
  I = Eigen::MatrixXd::Identity(SIZE_X, SIZE_X);

}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;

}



void KalmanFilter::Predict() {
  /**
    * predict the state
  */
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose()  + Q_;
}

void KalmanFilter::processUpdate(const VectorXd &y){
  MatrixXd Ht = H_.transpose();
  MatrixXd Si = (H_ * P_ * Ht + R_).inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
    * update the state by using Kalman Filter equations
  */

    VectorXd z_pred = H_ * x_;
    processUpdate((const VectorXd) z - z_pred);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /*
  update the state by using Extended Kalman Filter equations
  */
  VectorXd z_pred_EKF;

  float sqrt_pxy2 = (float) sqrt( ( x_[0]*x_[0] + x_[1]*x_[1] ));
  z_pred_EKF = VectorXd(3);
  z_pred_EKF[0] = sqrt_pxy2;
  z_pred_EKF[1] = atan2(x_[1],x_[0]); // atan2 takes returns always in the correct quadrant
  z_pred_EKF[2] = (x_[0]*x_[2] +  x_[1]*x_[3]) / sqrt_pxy2;

  processUpdate(z - z_pred_EKF);

}
