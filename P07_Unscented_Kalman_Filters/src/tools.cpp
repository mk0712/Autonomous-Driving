#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
  VectorXd rmse = VectorXd::Zero(4);
  if(estimations.size() != 0 && estimations.size() == ground_truth.size())
  {
    for (int i = 0; i < estimations.size(); ++i) {
      VectorXd error = estimations[i] - ground_truth[i];
      // Elevate to square:
      error = error.array() * error.array();
      rmse += error;
    }
    // Get the mean
    rmse = rmse / estimations.size();
    // calculate the sqrt
    rmse = rmse.array().sqrt();
  }
  else
  {
    std::cout << __LINE__ << "\tError in Calculate RMSE vector dimensions. " << std::endl;

  }
  return rmse;
}
