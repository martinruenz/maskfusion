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

#include <Utils/Resolution.h>
#include <Utils/Stopwatch.h>
#include <pangolin/utils/file_utils.h>
#include "../Core/FrameData.h"

#include "LogReader.h"

#include <cassert>
#include <zlib.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <stack>

class KlgLogReader : public LogReader {
 public:
  KlgLogReader(std::string file, bool flipColors);

  virtual ~KlgLogReader();

  void getNext();

  void getPrevious();

  int getNumFrames();

  bool hasMore();

  bool rewind();

  void fastForward(int frame);

  const std::string getFile();

  FrameDataPointer getFrameData();

  void setAuto(bool value);

  std::stack<int> filePointers;

 private:
  FrameDataPointer readFrame();
  FrameDataPointer currentDataPointer;

  cv::Mat depthDecompressionBuffer;
  cv::Mat rgbDecompressionBuffer;
  cv::Mat depthBuffer;
  cv::Mat rgbBuffer;
};
