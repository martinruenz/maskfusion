//  ================================================================
//  Created by Gregory Kramida on 7/4/17.
//  Copyright (c) 2017 Gregory Kramida
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//  ================================================================
#include "PangolinReader.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

PangolinReader::PangolinReader(const std::string& file, bool flipColors)
    : LogReader(file, flipColors), interface(pangolin::OpenVideo(file)), uri(file) {
  std::vector<pangolin::StreamInfo> streams = interface->Streams();
  desiredImageSize = cv::Size(Resolution::getInstance().width(), Resolution::getInstance().height());
  if (streams.size() != 2 || (streams[0].PixFormat().channel_bits[0] != 16 || streams[0].PixFormat().channels != 1) ||
      (streams[1].PixFormat().channel_bits[0] != 8 || streams[1].PixFormat().channels != 3)) {
    throw std::runtime_error("Need 2 streams: depth, 16-bit greyscale, and rgb, 3-channel 8-bit. Perhaps check the uri?");
  }

  imageBuffer = new unsigned char[interface->SizeBytes()];
  bufferCursor = imageBuffer;

  configureConversion(streams[0], resampleDepth, depthBuffer);
  configureConversion(streams[1], resampleRgb, rgbBuffer);
  interface->Start();
}

void PangolinReader::configureConversion(pangolin::StreamInfo& stream, bool& conversionUsageFlag, cv::Mat& buffer) {
  buffer = cv::Mat(stream.Height(), stream.Width(), stream.PixFormat().channels == 1 ? CV_16UC1 : CV_8UC3, bufferCursor);
  if (stream.Width() != static_cast<size_t>(Resolution::getInstance().width()) ||
      stream.Height() != static_cast<size_t>(Resolution::getInstance().height())) {
    conversionUsageFlag = true;
  } else {
    conversionUsageFlag = false;
  }
  bufferCursor += stream.SizeBytes();
}

static bool continue_showing = true;
void PangolinReader::getNext() {
  lastFrameRetrieved = true;
  FrameDataPointer result = std::make_shared<FrameData>();
  if (!initialized) {
    hasMore();  // sets hasMoreImages, initialized = true
  }
  if (hasMoreImages) {
    if (resampleDepth) {
      cv::Mat intermediate;
      cv::resize(depthBuffer, intermediate, desiredImageSize);
      intermediate.convertTo(result->depth, CV_32FC1, 0.001);
    } else {
      depthBuffer.convertTo(result->depth, CV_32FC1, 0.001);
    }
    if (resampleRgb) {
      cv::resize(rgbBuffer, result->rgb, desiredImageSize);
    } else {
      rgbBuffer.copyTo(result->rgb);
    }
    currentDataPointer = result;
    // TODO: remove debug block
    //		if(continue_showing){
    //			cv::imshow("color",data.rgb);
    //			cv::imshow("depth",data.depth);
    //			if(cv::waitKey(0) == 27)
    //				continue_showing = false;
    //		}
  }
}

int PangolinReader::getNumFrames() { return 0; }

bool PangolinReader::hasMore() {
  initialized = true;

  if (lastFrameRetrieved) {
    lastFrameRetrieved = false;
    hasMoreImages = interface->GrabNext(imageBuffer);
  }
  return hasMoreImages;
}

bool PangolinReader::rewind() {
  interface->Stop();
  interface.release();
  interface = pangolin::OpenVideo(this->uri);
  interface->Start();
  return true;
}

void PangolinReader::getPrevious() { throw std::runtime_error("operation not supported for PangolinReader"); }

void PangolinReader::fastForward(int frame) { throw std::runtime_error("operation not supported for PangolinReader"); }

const std::string PangolinReader::getFile() { return this->uri; }

void PangolinReader::setAuto(bool value) {}

PangolinReader::~PangolinReader() {
  interface->Stop();
  delete[] imageBuffer;
}

FrameDataPointer PangolinReader::getFrameData() { return currentDataPointer; }
