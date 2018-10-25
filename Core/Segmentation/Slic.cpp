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

#include "Slic.h"
//#include "spdlog/spdlog.h"

std::vector<cv::Vec3b> Slic::slicColors;

Slic::Slic(unsigned width, unsigned height, int spixelSize, /*float scale,*/ gSLICr::COLOR_SPACE colorSpace) {
  assert(spixelSize > 10 && spixelSize < 256);

  this->spixelSize = spixelSize;  // ceil(sqrtf(1.0f /  (scale * scale) ));
  spixelX = width / spixelSize;
  spixelY = height / spixelSize;
  spixelNum = spixelX * spixelY;
  spixelCounts.resize(spixelNum);

  // Settings
  slicSettings.img_size.x = width;
  slicSettings.img_size.y = height;
  slicSettings.no_segs = spixelNum;  // not mandatory, since "GIVEN_SIZE"
  slicSettings.spixel_size = spixelSize;
  slicSettings.coh_weight = 0.6f;
  slicSettings.no_iters = 5;

  slicSettings.color_space = colorSpace;
  slicSettings.seg_method = gSLICr::GIVEN_SIZE;  // or gSLICr::GIVEN_NUM for given number
  // NO, see top slicSettings.seg_method = gSLICr::GIVEN_NUM; // or gSLICr::GIVEN_NUM for given number
  slicSettings.do_enforce_connectivity = false;  // wheter or not run the enforce connectivity step

  engine = std::make_shared<gSLICr::engines::seg_engine_GPU>(slicSettings);
  slicInput = std::make_shared<gSLICr::UChar4Image>(slicSettings.img_size, true, true);
}

void Slic::setInputImage(const cv::Mat& inImg, bool swapRedBlue) {
  assert(inImg.type() == CV_8UC3);
  assert(inImg.isContinuous());
  inputImage = inImg;
  cv::Vec3b* inPtr = (cv::Vec3b*)inImg.data;
  gSLICr::Vector4u* slicPtr = slicInput->GetData(MEMORYDEVICE_CPU);
  if (swapRedBlue) {
#pragma omp parallel for
    for (size_t index = 0; index < slicInput->dataSize; index++) {
      slicPtr[index].r = inPtr[index][2];
      slicPtr[index].g = inPtr[index][1];
      slicPtr[index].b = inPtr[index][0];
    }
  } else {
#pragma omp parallel for
    for (size_t index = 0; index < slicInput->dataSize; index++) {
      slicPtr[index].r = inPtr[index][0];
      slicPtr[index].g = inPtr[index][1];
      slicPtr[index].b = inPtr[index][2];
    }
  }
}

void Slic::processFrame(bool countSizes) {
  engine->Perform_Segmentation(slicInput.get());
  slicResult = engine->Get_Seg_Mask();
  const int* slic = slicResult->GetData(MEMORYDEVICE_CPU);
  if (countSizes) {
    std::fill(spixelCounts.begin(), spixelCounts.end(), 0);
    for (unsigned index = 0; index < slicResult->dataSize; index++) spixelCounts[slic[index]]++;
  }
}

cv::Mat Slic::downsample() const {
  assert(slicSettings.color_space == gSLICr::RGB);  // Otherwise add conversion code

  const int* slic = getResult();

  gSLICr::Vector4u* inPtr = slicInput->GetData(MEMORYDEVICE_CPU);
  cv::Mat result = cv::Mat::zeros(spixelY, spixelX, CV_8UC3);
  cv::Vec3b* resData = (cv::Vec3b*)result.data;
  cv::Mat sums = cv::Mat::zeros(spixelY, spixelX, CV_32SC3);
  cv::Vec3i* sumData = (cv::Vec3i*)sums.data;
  unsigned empties = 0;
  for (unsigned indexBig = 0; indexBig < slicInput->dataSize; indexBig++)
    sumData[slic[indexBig]] += cv::Vec3i(inPtr[indexBig].b, inPtr[indexBig].g, inPtr[indexBig].r);

  for (unsigned index = 0; index < spixelNum; index++) {
    int cnt = spixelCounts[index];
    int readIndex = index;
    if (cnt == 0) {
      readIndex = resampleEmptyIndex(index);
      cnt = spixelCounts[readIndex];
      assert(cnt > 0);
      empties++;
    }
    resData[index] = cv::Vec3b(sumData[readIndex][0] / cnt, sumData[readIndex][1] / cnt, sumData[readIndex][2] / cnt);
  }

  // Optionally be verbose about empty superpixel.
  // if (empties != 0) std::cout << "Empty superpixel count: " << empties;

  return result;
}
