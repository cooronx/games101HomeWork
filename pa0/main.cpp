#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <iostream>

int main() {

  Eigen::Vector3f vec{2.0f, 1.0f, 1.0f};

  Eigen::Matrix3f mat;
  double angle = 45.0 / 180 * acos(-1);
  mat << cos(angle), -1 * sin(angle), 1, sin(angle), cos(angle), 2, 0, 0, 1;
  std::cout << "hello world" << std::endl;
  std::cout << mat * vec;
  return 0;
}