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

#include <vector>
#include <memory>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <gSLICr.h>

class Slic {
 public:
  Slic() {}
  Slic(unsigned width, unsigned height, int spixelSize, /*float scale,*/ gSLICr::COLOR_SPACE colorSpace = gSLICr::XYZ);

  virtual ~Slic() {}

  inline bool isValid() { return slicInput && engine; }
  inline unsigned getSuperpixelSize(unsigned index) const { return spixelCounts[index]; }

  // Returns CPU-Pointer of data (not expensive, no copying)
  inline const int* getResult() const { return slicResult->GetData(MEMORYDEVICE_CPU); }

  void setInputImage(const cv::Mat& inImg, bool swapRedBlue = true);
  void processFrame(bool countSizes = true);  // TODO refactor countSizes

  /// create downsampling of the input image
  cv::Mat downsample() const;

  /// create downsampling of other images (same scene as input image)
  template <typename T>
  cv::Mat downsample(const cv::Mat& image, int channel = -1) const {
    assert(image.isContinuous());
    assert(image.type() != CV_8UC1);  // Otherwise overflows are very likely
    assert(image.type() != cv::DataType<unsigned char>::type);
    assert(image.type() != cv::DataType<char>::type);
    assert(image.channels() == 1 || channel >= 0);

    cv::Mat result = cv::Mat::zeros(spixelY, spixelX, cv::DataType<T>::type);
    assert(result.channels() == 1);
    T* resData = (T*)result.data;
    T* inData = (T*)image.data;

    const int* slic = getResult();

    if (image.channels() == 1) {
      for (unsigned index = 0; index < image.total(); index++) resData[slic[index]] += inData[index];
    } else {
      assert(channel >= 0 && channel < image.channels());
      unsigned cIndex = channel;
      unsigned cStep = image.channels();
      for (unsigned index = 0; index < image.total(); index++) {
        resData[slic[index]] += inData[cIndex];
        cIndex += cStep;
      }
    }

    // Resample empty super-pixel, divide result by size of super-pixel
    for (unsigned index = 0; index < spixelNum; index++) {
      int cnt = spixelCounts[index];
      int readIndex = index;
      if (cnt == 0) {
        readIndex = resampleEmptyIndex(index);
        cnt = spixelCounts[readIndex];
        assert(cnt > 0);
      }
      resData[index] = resData[readIndex] / cnt;
    }

    return result;
  }

  /// create downsampling of other images (same scene as input image)
  template <typename T>
  cv::Mat downsampleThresholded(const cv::Mat& image, T minThreshold) const {
    assert(image.isContinuous());
    assert(image.type() != CV_8UC1);  // Otherwise overflows are very likely
    assert(image.type() != cv::DataType<unsigned char>::type);
    assert(image.type() != cv::DataType<char>::type);
    assert(image.channels() == 1);

    cv::Mat result = cv::Mat::zeros(spixelY, spixelX, cv::DataType<T>::type);
    T* resData = (T*)result.data;
    T* inData = (T*)image.data;

    const int* slic = getResult();
    std::vector<unsigned> downsampleCounts(result.total(), 0);

    for (unsigned index = 0; index < image.total(); index++) {
      if (inData[index] > minThreshold) {
        resData[slic[index]] += inData[index];
        downsampleCounts[slic[index]]++;
      }
    }

    // Resample empty super-pixel, divide result by size of super-pixel
    for (unsigned index = 0; index < spixelNum; index++) {
      int cnt = downsampleCounts[index];
      int readIndex = index;
      if (cnt == 0) {
        readIndex = resampleEmptyIndex(index);
        cnt = spixelCounts[readIndex];
        assert(cnt > 0);
      }
      resData[index] = resData[readIndex] / cnt;
    }

    return result;
  }

  template <typename T>
  cv::Mat upsample(const cv::Mat input) {
    return upsample<T, T>(input);
  }

  template <typename Tin, typename Tout>
  cv::Mat upsample(const cv::Mat input) {
    // assert(input.channels() == 1);
    assert(input.isContinuous());
    assert(input.total() == spixelNum);

    cv::Mat result = cv::Mat(slicInput->noDims[1], slicInput->noDims[0], cv::DataType<Tout>::type);
    Tout* resData = (Tout*)result.data;
    Tin* inData = (Tin*)input.data;

    const int* slic = getResult();
    for (unsigned index = 0; index < slicInput->dataSize; index++) resData[index] = (Tout)(inData[slic[index]]);

    return result;
  }

#ifdef SHOW_DEBUG_VISUALISATION
  inline void createSlicColors(unsigned num) {
    unsigned color = 0xa3b241;
    slicColors.reserve(num);
    for (unsigned i = slicColors.size(); i < num; i++) {
      color = color * (3 * i + i + 1);
      color = (color & (0xffffff)) >> 2;
      slicColors.push_back(cv::Vec3b((color & 0xff0000) >> 16, (color & 0x00ff00) >> 8, color & 0x0000ff));
    }
  }

  inline cv::Mat drawSurfelColors() {
    createSlicColors(spixelNum);
    cv::Mat res(slicResult->noDims.y, slicResult->noDims.x, CV_8UC3);
    const int* data = slicResult->GetData(MEMORYDEVICE_CPU);
    for (int y = 0; y < slicResult->noDims.y; y++)
      for (int x = 0; x < slicResult->noDims.x; x++) res.at<cv::Vec3b>(y, x) = slicColors[data[y * slicResult->noDims.x + x]];
    return res;
  }

  inline cv::Mat drawSurfelBorders(bool swapRedBlue) { return drawSurfelBorders(inputImage, swapRedBlue); }

  inline cv::Mat drawSurfelBorders(cv::Mat image, bool swapRedBlue) {
    cv::Mat res;
    if (image.type() == CV_8UC3)
      image.copyTo(res);
    else if (image.type() == CV_8UC1)
      cv::cvtColor(image, res, CV_GRAY2RGB);
    else
      assert(0);
    if (swapRedBlue) cv::cvtColor(res, res, cv::COLOR_RGB2BGR);
    const int* slic = slicResult->GetData(MEMORYDEVICE_CPU);
    for (int y = 1; y < slicResult->noDims.y - 1; y++) {
      for (int x = 1; x < slicResult->noDims.x - 1; x++) {
        int c = slic[y * slicResult->noDims.x + x];
        if (c != slic[y * slicResult->noDims.x + x + 1] || c != slic[y * slicResult->noDims.x + x - 1] ||
            c != slic[(y - 1) * slicResult->noDims.x + x + 1] || c != slic[(y + 1) * slicResult->noDims.x + x + 1])
          res.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
      }
    }
    return res;
  }
#endif

  inline cv::Point2i mapToHigh(int x, int y) const {
    return cv::Point2i(x * spixelSize + spixelSize * 0.5, y * spixelSize + spixelSize * 0.5);
  }

  inline cv::Point2i mapToHigh(int index) const { return mapToHigh(index % spixelX, index / spixelY); }

  inline int resampleEmptyIndex(unsigned index) const {
    cv::Point2i coord = mapToHigh(index);

    if (coord.y >= slicSettings.img_size.y) coord.y = slicSettings.img_size.y - 1;
    if (coord.x >= slicSettings.img_size.x) coord.x = slicSettings.img_size.x - 1;

    const int sampleIndex = coord.x + coord.y * slicSettings.img_size.x;
    int result = ((const int*)getResult())[sampleIndex];
    assert(result >= 0 && size_t(result) < spixelCounts.size());
    return result;
  }

  inline unsigned countSuperpixel() const { return spixelNum; }

 private:
  static std::vector<cv::Vec3b> slicColors;
  gSLICr::objects::settings slicSettings;
  std::shared_ptr<gSLICr::engines::seg_engine_GPU> engine;
  std::shared_ptr<gSLICr::UChar4Image> slicInput;
  cv::Mat inputImage;

  // Width*Height of original image, assigning super-pixel ID to each entry.
  // Gets updated when calling processFrame(), valid in between calls.
  const gSLICr::IntImage* slicResult;

  unsigned spixelNum;
  unsigned spixelSize;
  unsigned spixelX;
  unsigned spixelY;

  std::vector<unsigned> spixelCounts;  // TODO cv::Mat
};
