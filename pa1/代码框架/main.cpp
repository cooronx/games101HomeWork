#include "rasterizer.hpp"
#include <cmath>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos) {
  Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

  Eigen::Matrix4f translate;
  translate << 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1, -eye_pos[2],
      0, 0, 0, 1;

  view = translate * view;

  return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle) {
  Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

  // TODO: Implement this function
  // Create the model matrix for rotating the triangle around the Z axis.
  // Then return it.
  rotation_angle = rotation_angle / 180.0 * M_PI;
  model << std::cos(rotation_angle), -std::sin(rotation_angle), 0, 0,
      std::sin(rotation_angle), std::cos(rotation_angle), 0, 0, 0, 0, 1, 0, 0,
      0, 0, 1;
  return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar) {
  // Students will implement this function

  Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
  eye_fov = eye_fov / 180.0 * M_PI;
  zNear *= -1;
  zFar *= -1;
  projection << 1 / (aspect_ratio * std::tan(eye_fov / 2)), 0, 0, 0, 0,
      1 / (std::tan(eye_fov / 2)), 0, 0, 0, 0, (zNear + zFar) / (zNear - zFar),
      2 * zNear * zFar / (zNear - zFar), 0, 0, -1, 0;

  // float t = abs(zNear) * tan(eye_fov / 2.0 / 180.0 * MY_PI);
  // float b = -t;
  // float r = aspect_ratio * t;
  // float l = -r;

  // Matrix4f Orth_Thran, Orth_Scale;
  // Orth_Scale << 2 / (r - l), 0, 0, 0, 0, 2 / (t - b), 0, 0, 0, 0,
  //     2 / (zNear - zFar), 0, 0, 0, 0, 1;
  // Orth_Thran << 1, 0, 0, -(r + l) / 2, 0, 1, 0, -(t + b) / 2, 0, 0, 1,
  //     -(zNear + zFar) / 2, 0, 0, 0, 1;

  // Matrix4f pers;
  // pers << zNear, 0, 0, 0, 0, zNear, 0, 0, 0, 0, zNear + zFar, -zNear * zFar,
  // 0,
  //     0, 1, 0;

  // projection = Orth_Scale * Orth_Thran * pers * projection;
  // std::cout << projection;
  return projection;
}

Eigen::Matrix4f get_rotation(Eigen::Vector3f axis, float angle) {
  angle = angle / 180.0 * MY_PI;
  Eigen::Matrix3f helper = Eigen::Matrix3f::Identity();
  Eigen::Matrix3f cross_n = Eigen::Matrix3f::Identity();
  helper = axis * axis.transpose();
  cross_n << 0, -axis.z(), axis.y(), axis.z(), 0, -axis.x(), -axis.y(),
      axis.x(), 0;
  Eigen::Matrix3f ret_3d = std::cos(angle) * Eigen::Matrix3f::Identity() +
                           (1 - std::cos(angle)) * helper +
                           std::sin(angle) * cross_n;
  Eigen::Matrix4f ret_4d = Eigen::Matrix4f::Identity();
  ret_4d << ret_3d(0, 0), ret_3d(0, 1), ret_3d(0, 2), 0, ret_3d(1, 0),
      ret_3d(1, 1), ret_3d(1, 2), 0, ret_3d(2, 0), ret_3d(2, 1), ret_3d(2, 2),
      0, 0, 0, 0, 1;
  // std::cout << ret_4d;
  return ret_4d;
}

int main(int argc, const char **argv) {
  float angle = 0;
  bool command_line = false;
  std::string filename = "output.png";

  if (argc >= 3) {
    command_line = true;
    angle = std::stof(argv[2]); // -r by default
    if (argc == 4) {
      filename = std::string(argv[3]);
    } else
      return 0;
  }

  rst::rasterizer r(700, 700);

  Eigen::Vector3f eye_pos = {0, 0, 5};

  std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};

  std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

  auto pos_id = r.load_positions(pos);
  auto ind_id = r.load_indices(ind);

  int key = 0;
  int frame_count = 0;

  if (command_line) {
    r.clear(rst::Buffers::Color | rst::Buffers::Depth);

    r.set_model(get_model_matrix(angle));
    r.set_view(get_view_matrix(eye_pos));
    r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

    r.draw(pos_id, ind_id, rst::Primitive::Triangle);
    cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
    image.convertTo(image, CV_8UC3, 1.0f);

    cv::imwrite(filename, image);

    return 0;
  }
  Eigen::Vector4f tt{1, 1, -60, 1};
  std::cout << get_projection_matrix(45, 1, 0.1, 50) << "\n";
  while (key != 27) {
    r.clear(rst::Buffers::Color | rst::Buffers::Depth);

    r.set_model(get_rotation({0, 1, 0}, angle));
    r.set_view(get_view_matrix(eye_pos));
    r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

    r.draw(pos_id, ind_id, rst::Primitive::Triangle);

    cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
    image.convertTo(image, CV_8UC3, 1.0f);
    cv::imshow("image", image);
    key = cv::waitKey(10);

    std::cout << "frame count: " << frame_count++ << '\n';

    if (key == 'a') {
      angle += 10;
    } else if (key == 'd') {
      angle -= 10;
    }
  }

  return 0;
}
