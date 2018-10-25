/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 *
 * The use of the code within this file and all code within files that
 * make up the software that is ElasticFusion is permitted for
 * non-commercial purposes only.  The full terms and conditions that
 * apply to the code within this file are detailed within the LICENSE.txt
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/>
 * unless explicitly stated.  By downloading this file you agree to
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#pragma once

#include "../Shaders/Shaders.h"
#include "../Shaders/Uniform.h"
#include "../Shaders/Vertex.h"
#include "../GPUTexture.h"
#include "../Utils/Resolution.h"
#include "../Utils/Intrinsics.h"
#include <pangolin/gl/gl.h>

#include "Buffers.h"

class ModelProjection {
 public:
  ModelProjection();
  virtual ~ModelProjection();

  // Project: Vertex+Confidence, RGB+Time, Normals+Radii using Points
  void predictIndices(const Eigen::Matrix4f& pose, const int time, const OutputBuffer& model, const float depthCutoff, const int timeDelta);

  void renderDepth(const float depthCutoff);

  enum Prediction { ACTIVE, INACTIVE };

  // Project: Vertex+Confidence, RGB, Normals, time using splats
  void combinedPredict(const Eigen::Matrix4f& pose, const OutputBuffer& model, const float depthCutoff, const float confThreshold,
                       const int time, const int maxTime, const int timeDelta,
                       // const float unaryConfWeight,
                       ModelProjection::Prediction predictionType);

  void synthesizeDepth(const Eigen::Matrix4f& pose, const OutputBuffer& model, const float depthCutoff, const float confThreshold,
                       const int time, const int maxTime, const int timeDelta);

  GPUTexture* getSparseIndexTex() { return &sparseIndexTexture; }

  GPUTexture* getSparseVertConfTex() { return &sparseVertexConfTexture; }

  GPUTexture* getSparseColorTimeTex() { return &sparseColorTimeTexture; }

  GPUTexture* getSparseNormalRadTex() { return &sparseNormalRadTexture; }

  GPUTexture* getDrawTex() { return &drawTexture; }

  GPUTexture* getDepthTex() { return &depthTexture; }

  GPUTexture* getSplatImageTex() { return &splatColorTexture; }

  GPUTexture* getSplatVertexConfTex() { return &slpatVertexConfTexture; }

  GPUTexture* getSplatNormalTex() { return &slpatNormalTexture; }

  GPUTexture* getSplatTimeTex() { return &slpatTimeTexture; }

  GPUTexture* getOldImageTex() { return &oldImageTexture; }

  GPUTexture* getOldVertexTex() { return &oldVertexTexture; }

  GPUTexture* getOldNormalTex() { return &oldNormalTexture; }

  GPUTexture* getOldTimeTex() { return &oldTimeTexture; }

  static const int FACTOR;

 private:
  std::shared_ptr<Shader> indexProgram;
  pangolin::GlFramebuffer indexFrameBuffer;
  pangolin::GlRenderBuffer indexRenderBuffer;
  GPUTexture sparseIndexTexture;
  GPUTexture sparseVertexConfTexture;
  GPUTexture sparseColorTimeTexture;
  GPUTexture sparseNormalRadTexture;

  std::shared_ptr<Shader> drawDepthProgram;
  pangolin::GlFramebuffer drawFrameBuffer;
  pangolin::GlRenderBuffer drawRenderBuffer;
  GPUTexture drawTexture;

  std::shared_ptr<Shader> depthProgram;
  pangolin::GlFramebuffer depthFrameBuffer;
  pangolin::GlRenderBuffer depthRenderBuffer;
  GPUTexture depthTexture;

  std::shared_ptr<Shader> combinedProgram;
  pangolin::GlFramebuffer combinedFrameBuffer;
  pangolin::GlRenderBuffer combinedRenderBuffer;
  GPUTexture splatColorTexture;       // color of rendered splats (GL_RGBA, GL_UNSIGNED_BYTE)
  GPUTexture slpatVertexConfTexture;  // GL_RGBA32F
  GPUTexture slpatNormalTexture;      // GL_RGBA32F
  GPUTexture slpatTimeTexture;        // GL_LUMINANCE16UI_EXT

  pangolin::GlFramebuffer oldFrameBuffer;
  pangolin::GlRenderBuffer oldRenderBuffer;
  GPUTexture oldImageTexture;
  GPUTexture oldVertexTexture;
  GPUTexture oldNormalTexture;
  GPUTexture oldTimeTexture;
};
