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

#include "OpenNI2LiveReader.h"

#include <opencv2/imgproc/imgproc.hpp>

OpenNI2LiveReader::OpenNI2LiveReader(std::string file, bool flipColors) : LogReader(file, flipColors), lastFrameTime(-1), lastGot(-1) {
  std::cout << "Creating live capture... ";
  std::cout.flush();

  asus = new OpenNI2Interface(Resolution::getInstance().width(), Resolution::getInstance().height());

  depthReadBuffer = cv::Mat(Resolution::getInstance().height(), Resolution::getInstance().width(), CV_16UC1);

  if (!asus->ok()) {
    std::cout << "failed!" << std::endl;
    std::cout << asus->error();
  } else {
    std::cout << "success!" << std::endl;

    std::cout << "Waiting for first frame";
    std::cout.flush();

    int lastDepth = asus->latestDepthIndex.getValue();

    do {
      usleep(33333);
      std::cout << ".";
      std::cout.flush();
      lastDepth = asus->latestDepthIndex.getValue();
    } while (lastDepth == -1);

    std::cout << " got it!" << std::endl;
  }
}

OpenNI2LiveReader::~OpenNI2LiveReader() { delete asus; }

void OpenNI2LiveReader::getNext() {
  int lastDepth = asus->latestDepthIndex.getValue();

  assert(lastDepth != -1);

  int bufferIndex = lastDepth % OpenNI2Interface::numBuffers;

  if (bufferIndex == lastGot) {
    return;
  }

  if (lastFrameTime == asus->frameBuffers[bufferIndex].second) {
    return;
  }

  FrameDataPointer data = std::make_shared<FrameData>();
  data->allocateRGBD(Resolution::getInstance().width(), Resolution::getInstance().height());

  memcpy(depthReadBuffer.data, asus->frameBuffers[bufferIndex].first.first, Resolution::getInstance().numPixels() * 2);
  memcpy(data->rgb.data, asus->frameBuffers[bufferIndex].first.second, Resolution::getInstance().numPixels() * 3);

  lastFrameTime = asus->frameBuffers[bufferIndex].second;

  data->timestamp = lastFrameTime;
  depthReadBuffer.convertTo(data->depth, CV_32FC1, 0.001);

  if (flipColors) data->flipColors();

  currentDataPointer = data;
}

const std::string OpenNI2LiveReader::getFile() { return Parse::get().baseDir().append("live"); }

FrameDataPointer OpenNI2LiveReader::getFrameData() { return currentDataPointer; }

int OpenNI2LiveReader::getNumFrames() { return std::numeric_limits<int>::max(); }

bool OpenNI2LiveReader::hasMore() { return true; }

void OpenNI2LiveReader::setAuto(bool value) {
  asus->setAutoExposure(value);
  asus->setAutoWhiteBalance(value);
}
