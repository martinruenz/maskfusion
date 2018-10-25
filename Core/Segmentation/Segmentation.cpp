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

#include <list>
#include <tuple>

#include "Segmentation.h"
#include "PreSegmentation.h"
#include "../Model/Model.h"

SegmentationResult::ModelData::ModelData(unsigned t_id) : id(t_id) {}

SegmentationResult::ModelData::ModelData(unsigned t_id, ModelListIterator const& t_modelListIterator, cv::Mat const& t_lowICP,
                                         cv::Mat const& t_lowConf, unsigned t_superPixelCount, float t_avgConfidence)
    : id(t_id),
      modelListIterator(t_modelListIterator),
      lowICP(t_lowICP),
      lowConf(t_lowConf),
      superPixelCount(t_superPixelCount),
      avgConfidence(t_avgConfidence) {}

void Segmentation::init(int width, int height,
                        Method method,
                        const CameraModel& cameraIntrinsics,
                        std::shared_ptr<GPUTexture> textureRGB,
                        std::shared_ptr<GPUTexture> textureDepthMetric,
                        bool usePrecomputedMasks,
                        GlobalProjection* globalProjection,
                        std::queue<FrameDataPointer>* pQueue) {

  this->method = method;

  switch(method){
  case Method::MASK_FUSION:
      segmentationPerformer = std::make_unique<MfSegmentation>(width,
                                                                   height,
                                                                   cameraIntrinsics,
                                                                   !usePrecomputedMasks,
                                                                   textureRGB,
                                                                   textureDepthMetric,
                                                                   globalProjection,
                                                                   pQueue);
      break;
  case Method::CO_FUSION:
      segmentationPerformer = std::make_unique<CfSegmentation>(width, height);
      break;
  case Method::PRECOMPUTED:
      segmentationPerformer = std::make_unique<PreSegmentation>();
      break;
  default:
      throw std::runtime_error("Unknown segmentation method.");
  }
}

std::vector<std::pair<std::string, std::shared_ptr<GPUTexture> > > Segmentation::getDrawableTextures(){
    if(segmentationPerformer) return segmentationPerformer->getDrawableTextures();
    return std::vector<std::pair<std::string, std::shared_ptr<GPUTexture> > >();
}


SegmentationResult Segmentation::performSegmentation(std::list<std::shared_ptr<Model>>& models, FrameDataPointer frame,
                                                     unsigned char nextModelID, bool allowNew){
    //if (frame.mask.total()) performSegmentationPrecomputed(models, frame, nextModelID, allowNew);
    return segmentationPerformer->performSegmentation(models, frame, nextModelID, allowNew);
}

void Segmentation::cleanup(){
    segmentationPerformer.reset();
}
