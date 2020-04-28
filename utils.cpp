#include "utils.hpp"
using std::vector;

cv::Rect VecToRect(const vector<float> & vec)
{
  return cv::Rect(cv::Point(vec[1], vec[2]), cv::Point(vec[3], vec[4]));
}

void DrawRectangles(cv::Mat & img,
                    const vector<vector<float>> & vecVecFloat)
{
  for (const auto & vec: vecVecFloat)
    cv::rectangle(img, VecToRect(vec),  WHITE_COLOR);
}

void DrawRectangles(cv::Mat & img,
                    const vector<cv::Rect> & vecRect)
{
  for (const auto & rect: vecRect)
    cv::rectangle(img, rect,  WHITE_COLOR);
}