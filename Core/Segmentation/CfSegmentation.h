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

#include "Slic.h"
#include "SegmentationPerformer.h"

class Model;

class CfSegmentation : public SegmentationPerformer {
public:
  // Parameters
    const float MAX_DEPTH = 100;  // 100m
    unsigned crfIterations = 10;

    // CRF
    // pairwise
    float scaleFeaturesRGB = 1.0f / 30;
    float scaleFeaturesDepth = 1.0f / 0.4;  // TODO: AUTO
    float scaleFeaturesPos = 1.0f / 8;
    float weightAppearance = 40;
    float weightSmoothness = 40;
    // unary
    float unaryThresholdNew = 5;
    float unaryKError = 0.01;
    float unaryWeightError = 40;
    float unaryWeightErrorBackground = 10;
    float unaryWeightConfBackground = 0.1;

    CfSegmentation(int w, int h);
    virtual ~CfSegmentation();

    /// denseCRF Compute a segmentation of labels based on a fully connected CRF, using icp+projection unary terms and rgb+position+depth pairwise terms
    virtual SegmentationResult performSegmentation(std::list<std::shared_ptr<Model>>& models,
                                                   FrameDataPointer frame,
                                                   unsigned char nextModelID,
                                                   bool allowNew);

  inline void setPairwiseSigmaRGB(float v) { scaleFeaturesRGB = 1.0f / v; }
  inline void setPairwiseSigmaDepth(float v) { scaleFeaturesDepth = 1.0f / v; }
  inline void setPairwiseSigmaPosition(float v) { scaleFeaturesPos = 1.0f / v; }
  inline void setPairwiseWeightAppearance(float v) { weightAppearance = v; }
  inline void setPairwiseWeightSmoothness(float v) { weightSmoothness = v; }
  inline void setUnaryThresholdNew(float v) { unaryThresholdNew = v; }
  inline void setUnaryWeightError(float v) { unaryWeightError = v; }
  inline void setUnaryKError(float v) { unaryKError = v; }
  inline void setIterationsCRF(unsigned i) { crfIterations = i; }

 private:
  Slic slic;
  Eigen::MatrixXf lastRawCRF;
};
