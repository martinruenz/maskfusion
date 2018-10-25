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

#include <Utils/Resolution.h>
#include <Utils/Stopwatch.h>
#include <pangolin/utils/file_utils.h>

#include "LogReader.h"

#include <cassert>
#include <zlib.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <stack>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <atomic>

class ImageLogReader : public LogReader {
 public:
  ImageLogReader(std::string colorDirectory, std::string depthDirectory, std::string maskDirectory, unsigned indexWidth = 4,
                 std::string colorPrefix = "", std::string depthPrefix = "", std::string maskPrefix = "", bool flipColors = false);

  virtual ~ImageLogReader();
  void deleteFrames();  // TODO smart pointer

  void getNext();

  void getPrevious();

  int getNumFrames();

  bool hasMore();

  bool rewind();

  void fastForward(int frame);

  void bufferFrames();

  // load pair of depth image (like Depth0001.exr) and RGB image (Image0001.png/jpg) and optionally mask images
  // (Mask0001.pgm/png)
  FrameDataPointer loadFrameFromDrive(const size_t& index);

  const std::string getFile() { return file; }

  FrameDataPointer getFrameData();

  void setAuto(bool value) { assert(0); }

  inline void ignoreMask() { hasMasksGT = false; }
  void loadMaskIDs(const std::string &descrFile, std::vector<int>* outClassIds, std::vector<cv::Rect>* outROIs) const;
  inline void setMaxMasks(int v) { maxMasks = v; }
  inline bool hasPrecomputedMasksOnly() const { return hasMasks() && (int(maxMasks)==numFrames); } // TODO move to LogReader

 private:
  void startBufferLoop();
  void stopBufferLoop();
  void bufferLoop();
  void bufferFramesImpl();
  const int minBuffered = 30;
  size_t maxMasks = 0;  // Max number of masks that should be used (in order to initialise with ground truth, for tests)

  const std::string depthImagesDir;
  const std::string maskImagesDir;
  std::string colorExt;
  std::string depthExt;
  std::string maskExt;
  std::string colorPre;
  std::string depthPre;
  std::string maskPre;
  unsigned indexW = 4;
  unsigned startIndex = 0;

  // Data
  std::vector<FrameDataPointer> frames;
  unsigned nextBufferIndex = 0;
  float rateHz = 24;

  // BufferThread
  std::mutex bufferingMutex;
  std::condition_variable bufferingCondition;
  std::atomic<bool> bufferingLoopActive;
  std::thread bufferingThread;
};
