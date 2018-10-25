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
#pragma once

#include "LogReader.h"
#include <pangolin/pangolin.h>

class PangolinReader : public LogReader {
 public:
  PangolinReader(const std::string& file, bool flipColors);
  virtual ~PangolinReader();
  virtual void getNext();
  virtual void getPrevious();

  virtual int getNumFrames();

  virtual bool hasMore();

  virtual bool rewind();

  virtual void fastForward(int frame);

  virtual const std::string getFile();

  virtual void setAuto(bool value);

  virtual FrameDataPointer getFrameData();

 private:
  void configureConversion(pangolin::StreamInfo& stream,  // in
                           bool& conversionUsageFlag,     // out
                           cv::Mat& buffer                // out

                           );

  std::unique_ptr<pangolin::VideoInterface> interface;
  std::string uri;

  // buffers & conversion
  FrameDataPointer currentDataPointer;
  bool resampleDepth = false;
  bool resampleRgb = false;
  cv::Size desiredImageSize;

  cv::Mat depthBuffer;
  cv::Mat rgbBuffer;
  unsigned char* imageBuffer;
  unsigned char* bufferCursor;

  // playback
  bool lastFrameRetrieved = true;
  bool hasMoreImages = true;
  bool initialized = false;
};
