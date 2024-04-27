//
// Created by goksu on 4/6/19.
//

#include "rasterizer.hpp"
#include "Triangle.hpp"
#include <algorithm>
#include <cmath>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <vector>

float msaa_depth_buf[490005][4];

rst::pos_buf_id
rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions) {
  auto id = get_next_id();
  pos_buf.emplace(id, positions);

  return {id};
}

rst::ind_buf_id
rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices) {
  auto id = get_next_id();
  ind_buf.emplace(id, indices);

  return {id};
}

rst::col_buf_id
rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols) {
  auto id = get_next_id();
  col_buf.emplace(id, cols);

  return {id};
}

auto to_vec4(const Eigen::Vector3f &v3, float w = 1.0f) {
  return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

static bool insideTriangle(float x, float y, const Vector3f *_v) {
  // TODO : Implement this function to check if the point (x, y) is inside the
  // triangle represented by _v[0], _v[1], _v[2]
  //* 通过叉乘方向进行判断
  bool isInside = true;
  int sign = -1;
  for (int i = 0; i < 3; ++i) {
    Eigen::Vector3f now_v{_v[i].x(), _v[i].y(), 0};
    Eigen::Vector3f next_v{_v[(i + 1) % 3].x(), _v[(i + 1) % 3].y(), 0};
    //* 三角形边向量
    auto vec_v = next_v - now_v;
    //* 和要判断的点的向量
    Eigen::Vector3f p{x, y, 0};
    auto vec_p = p - now_v;
    auto cross_vec = vec_v.cross(vec_p);
    // std::cout << cross_vec << "\n";
    if (sign == -1) {
      sign = cross_vec.z() > 0 ? 1 : 0;
    } else {
      int now_sign = cross_vec.z() > 0 ? 1 : 0;
      if (now_sign != sign) {
        isInside = false;
        break;
      }
    }
  }
  return isInside;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y,
                                                            const Vector3f *v) {
  float c1 =
      (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y +
       v[1].x() * v[2].y() - v[2].x() * v[1].y()) /
      (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() +
       v[1].x() * v[2].y() - v[2].x() * v[1].y());
  float c2 =
      (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y +
       v[2].x() * v[0].y() - v[0].x() * v[2].y()) /
      (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() +
       v[2].x() * v[0].y() - v[0].x() * v[2].y());
  float c3 =
      (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y +
       v[0].x() * v[1].y() - v[1].x() * v[0].y()) /
      (v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() +
       v[0].x() * v[1].y() - v[1].x() * v[0].y());
  return {c1, c2, c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer,
                           col_buf_id col_buffer, Primitive type) {
  auto &buf = pos_buf[pos_buffer.pos_id];
  auto &ind = ind_buf[ind_buffer.ind_id];
  auto &col = col_buf[col_buffer.col_id];

  float f1 = (50 - 0.1) / 2.0;
  float f2 = (50 + 0.1) / 2.0;

  Eigen::Matrix4f mvp = projection * view * model;
  for (auto &i : ind) {
    Triangle t;
    Eigen::Vector4f v[] = {mvp * to_vec4(buf[i[0]], 1.0f),
                           mvp * to_vec4(buf[i[1]], 1.0f),
                           mvp * to_vec4(buf[i[2]], 1.0f)};
    // Homogeneous division
    for (auto &vec : v) {
      vec /= vec.w();
    }
    // Viewport transformation
    for (auto &vert : v) {
      vert.x() = 0.5 * width * (vert.x() + 1.0);
      vert.y() = 0.5 * height * (vert.y() + 1.0);
      vert.z() = vert.z() * f1 + f2;
    }

    for (int i = 0; i < 3; ++i) {
      t.setVertex(i, v[i].head<3>());
      t.setVertex(i, v[i].head<3>());
      t.setVertex(i, v[i].head<3>());
    }

    auto col_x = col[i[0]];
    auto col_y = col[i[1]];
    auto col_z = col[i[2]];

    t.setColor(0, col_x[0], col_x[1], col_x[2]);
    t.setColor(1, col_y[0], col_y[1], col_y[2]);
    t.setColor(2, col_z[0], col_z[1], col_z[2]);

    rasterize_triangle(t);
  }
}

//* 可变模板参数实现多个数字比较
template <typename T> T Maxs(const T &value) { return value; }
template <typename T, typename... Types>
T Maxs(const T &value, const Types &...args) {
  return std::max(value, Maxs(args...));
}

template <typename T> T Mins(const T &value) { return value; }
template <typename T, typename... Types>
T Mins(const T &value, const Types &...args) {
  return std::min(value, Mins(args...));
}

// Screen space rasterization
//* msaa2x 实现完成
void rst::rasterizer::rasterize_triangle(const Triangle &t) {
  auto v = t.toVector4();

  //* 包围盒
  int max_x = std::ceil(Maxs(t.v[0].x(), t.v[1].x(), t.v[2].x()));
  int min_x = std::floor(Mins(t.v[0].x(), t.v[1].x(), t.v[2].x()));
  int max_y = std::ceil(Maxs(t.v[0].y(), t.v[1].y(), t.v[2].y()));
  int min_y = std::floor(Mins(t.v[0].y(), t.v[1].y(), t.v[2].y()));

  for (int y = min_y; y <= max_y; ++y) {
    for (int x = min_x; x <= max_x; ++x) {
      int sample_idx = 0;
      int count = 0;
      for (float i = 0.25; i <= 0.75; i += 0.5) {
        for (float j = 0.25; j <= 0.75; j += 0.5) {
          ++sample_idx;
          if (insideTriangle(x + j, y + i, t.v)) {
            ++count;
            auto [alpha, beta, gamma] = computeBarycentric2D(x + j, y + i, t.v);
            float w_reciprocal =
                1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
            float z_interpolated = alpha * v[0].z() / v[0].w() +
                                   beta * v[1].z() / v[1].w() +
                                   gamma * v[2].z() / v[2].w();
            z_interpolated *= w_reciprocal;
            int idx = get_index(x, y);
            if (z_interpolated < msaa_depth_buf[idx][sample_idx]) {
              msaa_depth_buf[idx][sample_idx] = z_interpolated;
              Eigen::Vector3f color = frame_buf[idx];
              //* 按照插值比例计算原本像素的颜色
              frame_buf[idx] = ((count * 1.0 / 4) * t.getColor() +
                                (1.0 - (count * 1.0 / 4)) * color);
            }
          }
        }
      }
    }
  }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f &m) { model = m; }

void rst::rasterizer::set_view(const Eigen::Matrix4f &v) { view = v; }

void rst::rasterizer::set_projection(const Eigen::Matrix4f &p) {
  projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff) {
  if ((buff & rst::Buffers::Color) == rst::Buffers::Color) {
    std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
  }
  if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth) {
    std::fill(depth_buf.begin(), depth_buf.end(),
              std::numeric_limits<float>::infinity());
  }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h) {
  std::cout << "执行了一次"
            << "\n";
  frame_buf.resize(w * h);
  depth_buf.resize(w * h);
  for (int i = 0; i < 490002; ++i) {
    for (int j = 0; j < 4; ++j) {
      msaa_depth_buf[i][j] = std::numeric_limits<float>::infinity();
    }
  }
}

int rst::rasterizer::get_index(int x, int y) {
  return (height - 1 - y) * width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f &point,
                                const Eigen::Vector3f &color) {
  // old index: auto ind = point.y() + point.x() * width;
  auto ind = (height - 1 - point.y()) * width + point.x();
  frame_buf[ind] = color;
}

// clang-format on