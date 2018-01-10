#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
    // the R matrices represent the covariance error matrix, i.e. the noise for each value
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
                   0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
                0,  0.0009, 0,
                0,  0, 0.09;

  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
  /* Initial values of Hj appear to not influence the result */
  Hj_ << 10, 10, 0, 0,
         10, 10, 0, 0,
         10,  1,  1,  1;
  /**
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
    * Q is the process noise
    * R is the measurement noise
  */
  ekf_.F_ = MatrixXd(4,4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;
    noise_ax = 7.0;
    noise_ay = 9.0;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix. Which oneeee?
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd::Zero(4);
    ekf_.Q_ = MatrixXd(4, 4);
    // this values? TODO
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ <<  1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1000, 0,
                0, 0, 0,  1000;
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */

        // output the estimation in the cartesian coordinates
        float_t ro = measurement_pack.raw_measurements_(0);
        float_t phi = measurement_pack.raw_measurements_(1);
        float_t ro_dot = measurement_pack.raw_measurements_(2);
        float_t px = ro * cos(phi);
        float_t py = ro * sin(phi);
        if(fabs(px) <= 0.0001f || fabs(py) <= 0.0001f){
          return;
        }
        ekf_.x_ << px, py, 0, 0;

    }
    else /*if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) */{
      /**
      Initialize state.
      */
      if(fabs(measurement_pack.raw_measurements_[0]) <= 0.0001f || fabs(measurement_pack.raw_measurements_[1]) <= 0.0001f){
        return;
      }
        ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], \
                  0,0;
    }
    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
    float_t dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = measurement_pack.timestamp_;

  if(dt >= 0.001f)
  {

    float_t dt_2 = dt * dt;
    float_t dt_3 = dt_2 * dt;
    float_t dt_4 = dt_3 * dt;

    //Modify the F matrix so that the time is integrated
    // this updates the state transition function
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;

    //set the process covariance matrix Q
    // updates the process noise covariance matrix
    ekf_.Q_ <<  dt_4/4.0f*noise_ax, 0, dt_3/2.0f*noise_ax, 0,
            0, dt_4/4.0f*noise_ay, 0, dt_3/2.0f*noise_ay,
            dt_3/2.0f*noise_ax, 0, dt_2*noise_ax, 0,
            0, dt_3/2.0f*noise_ay, 0, dt_2*noise_ay;

    ekf_.Predict();
  }


  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
     * P state covariance matrix
     * H, Hj
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    // User EKF version, as radar is not linear
    // Calculate the jacobian
    Hj_ << Tools::CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_); // needs to be converted polar->cart
  } else {
    // Laser updates
    // use the linear version of the KF
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);

  }


}
