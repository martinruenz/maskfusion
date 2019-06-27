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

#include <MaskFusion.h>
#include <Utils/Parse.h>

#include "Tools/GUI.h"
#include "Tools/GroundTruthOdometry.h"
#include "Tools/LogReader.h"

class MainController {
 public:
  MainController(int argc, char* argv[]);
  virtual ~MainController();

  void launch();

 private:
  void run();

  enum DRAW_COLOR_TYPE { DRAW_USER_DEFINED = 0, DRAW_NORMALS = 1, DRAW_COLOR = 2, DRAW_TIMES = 3, DRAW_LABEL = 4 };
  void drawScene(DRAW_COLOR_TYPE backgroundColor = DRAW_USER_DEFINED, DRAW_COLOR_TYPE objectColor = DRAW_USER_DEFINED);
  void drawSphere(const float radius, const unsigned latLines = 10, const unsigned longLines = 10);
  void drawLineFromTo(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1, float lineWidth=2.5);

  void loadCalibration(const std::string& filename);

  bool good;
  MaskFusion* maskFusion;
  GUI* gui;
  bool showcaseMode;
  GroundTruthOdometry* groundTruthOdometry;
  std::unique_ptr<LogReader> logReader;

  bool iclnuim;
  std::string logFile;
  std::string poseFile;
  std::string exportDir;
  bool exportSegmentation;
  bool exportViewport;
  bool exportLabels;
  bool exportNormals;
  bool exportPoses;
  bool exportModels;

  float confGlobalInit, confObjectInit, icpErrThresh, covThresh, photoThresh, fernThresh;

  int timeDelta, icpCountThresh, start, end, preallocatedModelsCount, frameQueueSize;

  bool fillIn, openLoop, reloc, frameskip, quit, fastOdom, so3, rewind, frameToFrameRGB, usePrecomputedMasksOnly;

  int framesToSkip;
  bool streaming;
  bool resetButton;
  Segmentation::Method segmentationMethod;

  GPUResize* resizeStream;

  std::set<int> trackableClassIds;
};
