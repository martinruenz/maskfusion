/*
 * This file is part of https://github.com/martinruenz/maskfusion
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>
 */

#pragma once

#include <stdint.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <utility>
#include <memory>

struct FrameData {
  // Allocate memory for rgb and depth image
  void allocateRGBD(unsigned width, unsigned height) {
    rgb = cv::Mat(height, width, CV_8UC3);
    depth = cv::Mat(height, width, CV_32FC1);
  }

  int64_t timestamp = 0;
  int64_t index = 0;

  cv::Mat mask;   // External segmentation (optional!), CV_8UC1
  cv::Mat rgb;    // RGB data, CV_8UC3
  cv::Mat depth;  // Depth data, CV_32FC1

  std::vector<int> classIDs; // It is assumed that mask-labels are consecutive and that classIDs[mask.data[i]] provides the class for each pixel in the mask.
  std::vector<cv::Rect> rois;

  void flipColors() {
#pragma omp parallel for
    for (unsigned i = 0; i < rgb.total() * 3; i += 3) std::swap(rgb.data[i + 0], rgb.data[i + 2]);
  }
};

typedef std::shared_ptr<FrameData> FrameDataPointer;
