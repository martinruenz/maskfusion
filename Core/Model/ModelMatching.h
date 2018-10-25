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

#include <memory>

#include "Model.h"

class IModelMatcher {
 public:
  /// Try to detect one of the inactive models in the specified segmented region
  virtual ModelDetectionResult detectInRegion(const FrameData& frame, const cv::Rect& rect) = 0;

  /// Build model description. Returns true if succeeded else false (for instance, because the model is not having enough
  /// points to create descr.)
  virtual bool buildModelDescription(Model* model) = 0;
};

// [Removed matching code]
