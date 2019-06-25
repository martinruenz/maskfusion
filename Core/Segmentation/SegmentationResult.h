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
#include <Eigen/Core>
#include <opencv2/imgproc/imgproc.hpp>
#include <list>
#include <memory>
#include "../Utils/BoundingBox.h"

class Model;
typedef std::list<std::shared_ptr<Model>> ModelList;
typedef ModelList::iterator ModelListIterator;

// TODO Separate specific data (CRF) to somewhere else


struct SegmentationResult {

  // CV_8UC1, stores model-id per pixel
  cv::Mat fullSegmentation;

  bool hasNewLabel = false;
  float depthRange;

  // Optional
  cv::Mat lowCRF;
  cv::Mat lowRGB;
  cv::Mat lowDepth;

  struct ModelData {
    // Warning order makes a difference here!
    unsigned id; // FIXME this should not be part of the result
    ModelListIterator modelListIterator;

    cv::Mat lowICP;
    cv::Mat lowConf;

    bool isNonStatic = false;
    bool isEmpty = true;
    unsigned superPixelCount = 0; // TODO refactor this
    unsigned pixelCount = 0;
    float avgConfidence = 0;
    int classID = -1;

    float depthMean = 0;
    float depthStd = 0;

    // The following values are only approximations:
    BoundingBox boundingBox;

    // Required for partially supported C++14 (in g++ 4.9.4)
    ModelData(unsigned t_id);

    ModelData(unsigned t_id, ModelListIterator const& t_modelListIterator, cv::Mat const& t_lowICP, cv::Mat const& t_lowConf,
              unsigned t_superPixelCount = 0, float t_avgConfidence = 0);
  };
  std::vector<ModelData> modelData;
};
