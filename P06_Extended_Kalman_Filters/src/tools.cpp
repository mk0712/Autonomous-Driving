#include <iostream>
#include "tools.h"
#include <cmath>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

 VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

    VectorXd rmse(4);
    rmse << 0,0,0,0;

    if (estimations.size() != 0 && estimations.size() == ground_truth.size()) {


        //accumulate squared residuals
        for (int i = 0; i < estimations.size(); ++i) {

            VectorXd error = estimations[i] - ground_truth[i];

            //coefficient-wise multiplication
            error = error.array() * error.array();
            rmse += error;
        }

        //calculate the mean
        rmse = rmse / estimations.size();

        //calculate the squared root
        rmse = rmse.array().sqrt();
    } else
    {
        std::cout << "Error in estimation or ground_thruth vector dimesions" << std::endl;
    }
    //return the result
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */
    MatrixXd Hj= MatrixXd::Zero(3,4);
    //recover state parameters
    float px = (float)x_state(0);
    float py = (float)x_state(1);
    float vx = (float)x_state(2);
    float vy = (float)x_state(3);


    float pxy2 = px*px + py*py;
    float sqrt_pxy2 =(float) sqrt(pxy2);


    //check division by zero
    if(std::fabs(pxy2) <= 0.0001f){
        std::cout << "Division by 0 in TOOLS::CalculateJacobian" << std::endl;
    }
    else{
    //compute the Jacobian matrix
      Hj(0,0) = px / sqrt_pxy2;
      Hj(0,1) = py / sqrt_pxy2;
      Hj(1,0) = (-1.0f * py)/pxy2;
      Hj(1,1) = px/pxy2;
      Hj(2,0) = py * (vx*py - vy*px)/(pxy2*sqrt_pxy2);
      Hj(2,1) = px * (vy*px - vx*py)/(pxy2*sqrt_pxy2);
      Hj(2,2) = px / sqrt_pxy2;
      Hj(2,3) = py / sqrt_pxy2;
    }

    return Hj;
}
