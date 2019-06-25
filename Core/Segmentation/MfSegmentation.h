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

#include "../Cuda/containers/device_array.hpp"
#include "../Cuda/types.cuh"
#include "../FrameData.h"
#include "SegmentationPerformer.h"

#include <list>
#include <queue>
#include <map>
#include <memory>

#include <pangolin/gl/gl.h>
#include "../Shaders/Shaders.h"
#include "../GPUTexture.h"

class GPUTexture;
class MaskRCNN;
class GlobalProjection;

class MfSegmentation : public SegmentationPerformer {
public:

 // Parameters
  float bilatSigmaDepth = 3;
  float bilatSigmaColor = 8;
  float bilatSigmaLocation = 2;
  int bilatSigmaRadius = 2;

  float nonstaticThreshold = 0.4;
  float threshold = 0.1;
  float weightConvexity = 1;
  float weightDistance = 1;
  int morphEdgeIterations = 3;
  int morphEdgeRadius = 1;
  int morphMaskIterations = 3;
  int morphMaskRadius = 1;

  bool removeEdges = true; // These two are exclusive (otherwise wasting computations)
  bool removeEdgeIslands = false;

  const float minMaskModelOverlap; // Min overlap to existing models, ratio of mask size
  const int minMappedComponentSize;

  int personClassID = 255;

 public:
  MfSegmentation(int w,
                 int h,
                 const CameraModel& cameraIntrinsics,
                 bool embedMaskRCNN,
                 std::shared_ptr<GPUTexture> textureRGB,
                 std::shared_ptr<GPUTexture> textureDepthMetric,
                 GlobalProjection* globalProjection,
                 std::queue<FrameDataPointer>* queue = nullptr);
  virtual ~MfSegmentation();


  virtual SegmentationResult performSegmentation(std::list<std::shared_ptr<Model>>& models,
                                                 FrameDataPointer frame,
                                                 unsigned char nextModelID,
                                                 bool allowNew);

  virtual std::vector<std::pair<std::string, std::shared_ptr<GPUTexture>>> getDrawableTextures();

  void computeLookups();

 private:

  void allocateModelBuffers(unsigned char numModels);

  std::unique_ptr<MaskRCNN> maskRCNN;
  bool sequentialMaskRCNN;

  // Buffers for output / visualisation
  //DeviceArray2D<float> segmentationMap;
  std::shared_ptr<GPUTexture> segmentationMap;
  //std::shared_ptr<GPUTexture> debugMap;

  // CPU buffers for internal use
  cv::Mat cv8UC1Buffer;
  cv::Mat cvLabelComps;
  cv::Mat cvLabelEdges;
  struct ModelBuffers {
    unsigned int maskOverlap[256];
    unsigned char modelID;
  };
  std::vector<ModelBuffers> modelBuffers;
  unsigned char maskToID[256];
  unsigned char modelIDToIndex[256];
  unsigned char modelIndexToID[256];
  cv::Mat semanticIgnoreMap;

  // Buffers for internal use
  DeviceArray2D<float> floatEdgeMap;
  DeviceArray2D<float> floatBuffer;
  DeviceArray2D<unsigned char> binaryEdgeMap;
  DeviceArray2D<unsigned char> ucharBuffer;

  // 3D-data used for tracking is filtered too heavily, hence recompute here
  const bool REUSE_FILTERED_MAPS;
  DeviceArray2D<float> vertexMap;
  DeviceArray2D<float> normalMap;
  DeviceArray2D<float> depthMapMetric;
  DeviceArray2D<float> depthMapMetricFiltered;
  DeviceArray2D<uchar4> rgb;
  std::shared_ptr<GPUTexture> textureDepthMetric;
  std::shared_ptr<GPUTexture> textureRGB;
  CameraModel cameraIntrinsics;

  const GlobalProjection* globalProjection;
};
