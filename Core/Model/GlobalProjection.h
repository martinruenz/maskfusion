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

#include <list>

#include "../Shaders/Shaders.h"
#include "../Shaders/Uniform.h"
#include "../Shaders/Vertex.h"
#include "../GPUTexture.h"
#include "../Utils/Resolution.h"
#include "../Utils/Intrinsics.h"
#include <pangolin/gl/gl.h>

#include "Buffers.h"

class Model;

class GlobalProjection {
 public:
  GlobalProjection(int w, int h);
  virtual ~GlobalProjection();

  void project(const std::list<std::shared_ptr<Model>>& models, int time, int maxTime, int timeDelta, float depthCutoff);

  void downloadDirect(); // Speed: This could be accelerated with PBOs

  inline cv::Mat getProjectedModelIDs() const { return idBuffer; }
  inline cv::Mat getProjectedDepth() const { return depthBuffer; }

 private:

  std::shared_ptr<Shader> program;
  pangolin::GlFramebuffer framebuffer;
  pangolin::GlRenderBuffer renderbuffer;
  GPUTexture texDepth; // GL_R16F
  GPUTexture texID; // GL_R8UI

  cv::Mat depthBuffer;
  cv::Mat idBuffer;
};
