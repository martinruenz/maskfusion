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

#include "../Cuda/types.cuh"
#include "Slic.h"
#include "SegmentationPerformer.h"
#include "MfSegmentation.h"
#include "CfSegmentation.h"
#include <Eigen/Core>
#include <thread>
#include <opencv2/imgproc/imgproc.hpp>
#include <map>

class Model;
class GPUTexture;

class Segmentation {
 public:
  enum class Method { CO_FUSION, MASK_FUSION, PRECOMPUTED };

  void init(int width,
            int height,
            Method method,
            const CameraModel& cameraIntrinsics,
            std::shared_ptr<GPUTexture> textureRGB,
            std::shared_ptr<GPUTexture> depthMetric,
            bool usePrecomputedMasks,
            GlobalProjection* globalProjection,
            std::queue<FrameDataPointer>* pQueue = NULL);

  std::vector<std::pair<std::string, std::shared_ptr<GPUTexture> > > getDrawableTextures();

  SegmentationResult performSegmentation(std::list<std::shared_ptr<Model>>& models, FrameDataPointer frame, unsigned char nextModelID,
                                         bool allowNew);

//  SegmentationResult performSegmentationPrecomputed(std::list<std::shared_ptr<Model>>& models, FrameDataPointer frame, unsigned char nextModelID,
//                                         bool allowNew);

  MfSegmentation* getMfSegmentationPerformer() { return dynamic_cast<MfSegmentation*>(segmentationPerformer.get()); }
  CfSegmentation* getCfSegmentationPerformer() { return dynamic_cast<CfSegmentation*>(segmentationPerformer.get()); }
  SegmentationPerformer* getSegmentationPerformer() { return segmentationPerformer.get(); }

  void cleanup();

 private:

  // General
  Method method;

  std::unique_ptr<SegmentationPerformer> segmentationPerformer;
};
