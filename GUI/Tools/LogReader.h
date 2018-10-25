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

#include <string>
#include <zlib.h>
#include <poll.h>
#include <Utils/Img.h>
#include <Utils/Resolution.h>
#include "../../Core/FrameData.h"
#include "../../Core/Utils/Intrinsics.h"

#include "JPEGLoader.h"

class LogReader {
 public:
  LogReader(std::string file, bool flipColors)
      : flipColors(flipColors),
        currentFrame(0),
        file(file),
        width(Resolution::getInstance().width()),
        height(Resolution::getInstance().height()),
        numPixels(width * height) {}

  virtual ~LogReader() {}

  virtual void getNext() = 0;

  virtual int getNumFrames() = 0;

  virtual bool hasMore() = 0;

  virtual bool rewind() = 0;

  virtual void getPrevious() = 0;

  virtual void fastForward(int frame) = 0;

  virtual const std::string getFile() = 0;

  virtual void setAuto(bool value) = 0;

  bool flipColors;

  virtual FrameDataPointer getFrameData() = 0;

  virtual bool hasIntrinsics() const { return calibrationFile != ""; }
  virtual std::string getIntinsicsFile() const { return calibrationFile; }

  virtual bool hasMasks() const { return hasMasksGT; }

  int currentFrame;

 protected:
  int32_t depthImageSize;
  int32_t rgbImageSize;

  const std::string file;
  FILE* fp;
  int numFrames;
  int width;
  int height;
  int numPixels;

  std::string calibrationFile = "";
  bool hasMasksGT = false;

  JPEGLoader jpeg;
};
