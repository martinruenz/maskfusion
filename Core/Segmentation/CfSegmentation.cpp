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

#include <list>
#include <tuple>

#include "CfSegmentation.h"
#include "ConnectedLabels.hpp"
#include "densecrf.h"
#include "../Model/Model.h"

#ifdef SHOW_DEBUG_VISUALISATION
#include <iomanip>
#include "../Utils/Gnuplot.h"
#endif

#ifndef TICK
#define TICK(name)
#define TOCK(name)
#endif

#ifdef PRINT_CRF_TIMES
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::system_clock::time_point TimePoint;
#endif

CfSegmentation::CfSegmentation(int w, int h){
    slic = Slic(w, h, 16, gSLICr::RGB);
}

CfSegmentation::~CfSegmentation() {}

//SegmentationResult CfSegmentation::performSegmentationCRF(std::list<std::shared_ptr<Model>>& models, FrameDataPointer frame,
//                                                        unsigned char nextModelID, bool allowNew) {

SegmentationResult CfSegmentation::performSegmentation(std::list<std::shared_ptr<Model> > &models,
                                                           FrameDataPointer frame,
                                                           unsigned char nextModelID,
                                                           bool allowNew){

  assert(models.size() < 256);

  static unsigned CFRAME = 0;
  CFRAME++;

  TICK("SLIC+SCALING");

  SegmentationResult result;

  const unsigned numExistingModels = models.size();
  const unsigned numLabels = allowNew ? numExistingModels + 1 : numExistingModels;

  slic.setInputImage(frame->rgb);
  slic.processFrame();

  // TODO Speedup! See notes!!
  result.lowRGB = slic.downsample();
  result.lowDepth = slic.downsampleThresholded<float>(frame->depth, 0.02);

  assert(result.lowDepth.total() == result.lowRGB.total());

  const unsigned lowWidth = result.lowRGB.cols;
  const unsigned lowHeight = result.lowRGB.rows;
  const unsigned lowTotal = result.lowRGB.total();
  const unsigned fullTotal = frame->rgb.total();
  const int fullWidth = frame->rgb.cols;
  const int fullHeight = frame->rgb.rows;

#ifdef SHOW_DEBUG_VISUALISATION
  struct LabelDebugImages {
    cv::Mat vertConfTex;
    cv::Mat icpFull;
    cv::Mat icpLow;
    cv::Mat crfImage;
  };
  std::vector<LabelDebugImages> labelDebugImages;
#endif

  // Compute depth range
  float depthMin = std::numeric_limits<float>::max();
  float depthMax = 0;
  for (unsigned i = 0; i < result.lowDepth.total(); i++) {
    float d = ((float*)result.lowDepth.data)[i];
    if (d > MAX_DEPTH || d < 0 || !std::isfinite(d)) {
      // assert(0);
      continue;
    }
    if (depthMax < d) depthMax = d;
    if (depthMin > d) depthMin = d;
  }
  result.depthRange = depthMax - depthMin;

  // Compute per model data (ICP texture..)
  unsigned char modelIdToIndex[256];
  unsigned char mIndex = 0;
  for (auto it = models.begin(); it != models.end(); it++) {
    auto& m = *it;

    cv::Mat vertConfTex = m->downloadVertexConfTexture();
    cv::Mat icpFull = m->downloadICPErrorTexture();
    cv::Mat icp = slic.downsample<float>(icpFull);
    cv::Mat conf = slic.downsample<float>(vertConfTex, 3);
    result.modelData.push_back({m->getID(), it, icp, conf});
    modelIdToIndex[m->getID()] = mIndex++;

    // Average confidence
    auto& modelData = result.modelData.back();
    float highestConf = 0;
    for (unsigned j = 0; j < lowTotal; j++) {
      float& c = ((float*)modelData.lowConf.data)[j];
      if (!std::isfinite(c)) {
        c = 0;
        continue;
      }
      if (highestConf < c) highestConf = c;
      modelData.avgConfidence += c;
    }
    modelData.avgConfidence /= lowTotal;

#ifdef SHOW_DEBUG_VISUALISATION
    labelDebugImages.push_back({vertConfTex, icpFull, icp});
#endif
  }
  if (allowNew) {
    modelIdToIndex[nextModelID] = mIndex;
    result.modelData.push_back({nextModelID});

#ifdef SHOW_DEBUG_VISUALISATION
    labelDebugImages.push_back({cv::Mat(), cv::Mat(), cv::Mat::zeros(lowHeight, lowWidth, CV_32FC1)});
#endif
  }

  TOCK("SLIC+SCALING");
  TICK("CRF-FULL");

  DenseCRF2D crf(lowWidth, lowHeight, numLabels);
  Eigen::MatrixXf unary(numLabels, lowTotal);
  cv::Mat unaryMaxLabel(lowHeight, lowWidth, CV_8UC1);
  unsigned char* pUnaryMaxLabel = unaryMaxLabel.data;

  // auto clamp = [](float val, float min, float max) -> float {
  //    return std::max(std::min(val,max), min);
  //};
  auto getError = [&result](unsigned char modelIndex, unsigned pixelIndex) -> float& {
    return ((float*)result.modelData[modelIndex].lowICP.data)[pixelIndex];
  };
  auto getDepth = [&result](unsigned pixelIndex) -> float& { return ((float*)result.lowDepth.data)[pixelIndex]; };
  auto getConf = [&result](unsigned char modelIndex, unsigned pixelIndex) -> float& {
    return ((float*)result.modelData[modelIndex].lowConf.data)[pixelIndex];
  };

  for (unsigned k = 0; k < lowTotal; k++) {
    // Special case: If no label supplies confidence, prefer background
    float confSum = 0;
    for (unsigned i = 0; i < models.size(); i++) confSum += getConf(i, k);

    if (getConf(0, k) < 0.3) getError(0, k) = result.depthRange * 0.01;
    for (unsigned i = 1; i < numExistingModels; i++) {
      float& error = getError(i, k);
      float& conf = getConf(i, k);
      assert(std::isfinite(error));

      if (conf <= 0.4) {
        error = result.depthRange * unaryKError;
      }
    }

    // Depth sorting & updating of confidences
    // const float maxConf = 50;
    std::multimap<float, unsigned> sortedLabels;
    for (unsigned i = 0; i < numExistingModels; i++) {
      float d = getDepth(k);
      if (d > MAX_DEPTH)
        d = MAX_DEPTH;
      else if (d < 0)
        d = 0;
      sortedLabels.emplace(d, i);
    }

    // Object probabilities
    unsigned i;
    float sum = 0;
    float lowestError = getError(0, k) / result.depthRange;  // std::numeric_limits<float>::max();
    for (auto& l : sortedLabels) {
      i = l.second;
      const SegmentationResult::ModelData& modelData = result.modelData[i];

      float error = ((float*)modelData.lowICP.data)[k];
      assert(std::isfinite(error) && error >= 0);
      error /= result.depthRange;
      if (error < lowestError) lowestError = error;

      unary(i, k) = unaryWeightError * error;
      sum += unary(i, k);
    }

    if (allowNew) {
      unary(models.size(), k) = std::max(unaryThresholdNew - unaryWeightError * lowestError, 0.01f);
      sum += unary(models.size(), k);
    }

    unsigned char maxIndex = 0;
    float maxVal = 0;
    for (unsigned char i = 0; i < numLabels; i++) {
      // Find max label
      if (maxVal < unary(i, k)) {
        maxVal = unary(i, k);
        maxIndex = i;
      }
    }
    pUnaryMaxLabel[k] = maxIndex;

  }  // unaries

  // Make borders uncertain
  if (false) {
    for (unsigned y = 1; y < lowHeight - 1; y++) {
      for (unsigned x = 1; x < lowWidth - 1; x++) {
        const unsigned idx = y * lowWidth + x;
        unsigned char l = pUnaryMaxLabel[idx];
        // if(pUnaryMaxLabel[idx] == 0){
        if (pUnaryMaxLabel[idx - 1] != l || pUnaryMaxLabel[idx + 1] != l || pUnaryMaxLabel[idx - lowWidth] != l ||
            pUnaryMaxLabel[idx + lowWidth] != l || pUnaryMaxLabel[idx - lowWidth - 1] != l || pUnaryMaxLabel[idx - lowWidth + 1] != l ||
            pUnaryMaxLabel[idx - lowWidth - 1] != l || pUnaryMaxLabel[idx - lowWidth + 1] != l) {
          assert(0);  // see -log
          for (unsigned char i = 0; i < numLabels; i++) unary(i, idx) = -log(1.0f / numLabels);
        }
      }
    }
  }

#ifdef SHOW_DEBUG_VISUALISATION  // Visualise unary potentials

  auto floatToUC3 = [](cv::Mat image, float min = 0, float max = 1) -> cv::Mat {
    float range = max - min;
    cv::Mat tmp(image.rows, image.cols, CV_8UC3);
    for (unsigned i = 0; i < image.total(); i++) {
      unsigned char v = (std::min(image.at<float>(i), max) - min) / range * 255;
      tmp.at<cv::Vec3b>(i) = cv::Vec3b(v, v, v);
      // tmp.at<unsigned char>(i) = (std::min(image.at<float>(i), max) - min) / range * 255;
    }
    return tmp;
  };

  // Convert CRF-Matrix to Mat, whilst color-coding values (0..1) using COLORMAP_HOT
  auto mapCRFToImage = [&](Eigen::MatrixXf& field, unsigned index, int valueScale = 0 /* 0 linear, 1 log, 2 exp*/) -> cv::Mat {

    cv::Mat r;

    std::function<float(float)> mapVal;
    switch (valueScale) {
      case 1:
        mapVal = [](float v) -> float { return -log(v + 1); };
        break;
      case 2:
        mapVal = [](float v) -> float { return exp(-v); };
        break;
      default:
        mapVal = [](float v) -> float { return v; };
        break;
    }

    // Create greyscale image
    cv::Mat greyscale(lowHeight, lowWidth, CV_8UC1);
    unsigned char* pGreyscale = greyscale.data;
    for (unsigned k = 0; k < lowTotal; k++) pGreyscale[k] = mapVal(field(index, k)) * 255;

    // Create heat-map
    cv::applyColorMap(greyscale, r, cv::COLORMAP_HOT);

    // Find invalid values and highlight
    cv::Vec3b* pResult = (cv::Vec3b*)r.data;
    for (unsigned k = 0; k < lowTotal; k++) {
      float v = mapVal(field(index, k));
      if (v < 0)
        pResult[k] = cv::Vec3b(255, 0, 0);
      else if (v > 1)
        pResult[k] = cv::Vec3b(0, 255, 0);
    }

    return r;
  };

  const unsigned char colors[31][3] = {
      {0, 0, 0},     {0, 0, 255},     {255, 0, 0},   {0, 255, 0},     {255, 26, 184},  {255, 211, 0},   {0, 131, 246},  {0, 140, 70},
      {167, 96, 61}, {79, 0, 105},    {0, 255, 246}, {61, 123, 140},  {237, 167, 255}, {211, 255, 149}, {184, 79, 255}, {228, 26, 87},
      {131, 131, 0}, {0, 255, 149},   {96, 0, 43},   {246, 131, 17},  {202, 255, 0},   {43, 61, 0},     {0, 52, 193},   {255, 202, 131},
      {0, 43, 96},   {158, 114, 140}, {79, 184, 17}, {158, 193, 255}, {149, 158, 123}, {255, 123, 175}, {158, 8, 0}};
  auto getColor = [&colors](unsigned index) -> cv::Vec3b {
    return (index == 255) ? cv::Vec3b(255, 255, 255) : (cv::Vec3b)colors[index % 31];
  };

  auto mapLabelToColorImage = [&getColor](cv::Mat input, bool white0 = false) -> cv::Mat {

    std::function<cv::Vec3b(unsigned)> getIndex;
    auto getColorWW = [&](unsigned index) -> cv::Vec3b { return (white0 && index == 0) ? cv::Vec3b(255, 255, 255) : getColor(index); };

    if (input.type() == CV_32SC1)
      getIndex = [&](unsigned i) -> cv::Vec3b { return getColorWW(input.at<int>(i)); };
    else if (input.type() == CV_8UC1)
      getIndex = [&](unsigned i) -> cv::Vec3b { return getColorWW(input.data[i]); };
    else
      assert(0);
    cv::Mat result(input.rows, input.cols, CV_8UC3);
    for (unsigned i = 0; i < result.total(); ++i) {
      ((cv::Vec3b*)result.data)[i] = getIndex(i);
    }
    return result;
  };

  auto showInputOverlay = [&](cv::Mat original, cv::Mat segmentation) -> cv::Mat {
    assert(original.type() == CV_8UC3);
    cv::Mat result(original.rows, original.cols, CV_8UC3);
    cv::Vec3b* pResult = ((cv::Vec3b*)result.data);
    cv::Vec3b* pOriginal = ((cv::Vec3b*)original.data);
    cv::Mat overlay = mapLabelToColorImage(segmentation, true);
    cv::Vec3b* pOverlay = ((cv::Vec3b*)overlay.data);
    for (unsigned i = 0; i < result.total(); i++) pResult[i] = 0.85 * pOverlay[i] + 0.15 * pOriginal[i];
    return result;
  };

  auto stackImagesHorizontally = [](std::vector<cv::Mat> images) -> cv::Mat {
    if (images.size() == 0) return cv::Mat();
    unsigned totalWidth = 0;
    unsigned currentCol = 0;
    for (cv::Mat& m : images) totalWidth += m.cols;
    cv::Mat result(images[0].rows, totalWidth, images[0].type());
    for (cv::Mat& m : images) {
      m.copyTo(result(cv::Rect(currentCol, 0, m.cols, m.rows)));
      currentCol += m.cols;
    }
    return result;
  };

  auto stackImagesVertically = [](std::vector<cv::Mat> images) -> cv::Mat {
    if (images.size() == 0) return cv::Mat();
    unsigned totalHeight = 0;
    unsigned currentRow = 0;
    for (cv::Mat& m : images) totalHeight += m.rows;
    cv::Mat result(totalHeight, images[0].cols, images[0].type());
    for (cv::Mat& m : images) {
      m.copyTo(result(cv::Rect(0, currentRow, m.cols, m.rows)));
      currentRow += m.rows;
    }
    return result;
  };

  for (unsigned i = 0; i < numLabels; ++i) labelDebugImages[i].crfImage = mapCRFToImage(unary, i, 2);
#endif

  crf.setUnaryEnergy(unary);
  crf.addPairwiseGaussian(2, 2, new PottsCompatibility(weightSmoothness));

  Eigen::MatrixXf feature(6, lowTotal);
  for (unsigned j = 0; j < lowHeight; j++)
    for (unsigned i = 0; i < lowWidth; i++) {
      unsigned index = j * lowWidth + i;
      feature(0, index) = i * scaleFeaturesPos;                              // sx
      feature(1, index) = j * scaleFeaturesPos;                              // sy
      feature(2, index) = frame->rgb.data[index * 3 + 0] * scaleFeaturesRGB;  // sr
      feature(3, index) = frame->rgb.data[index * 3 + 1] * scaleFeaturesRGB;  // sg
      feature(4, index) = frame->rgb.data[index * 3 + 2] * scaleFeaturesRGB;  // sb
      feature(5, index) = std::min(((float*)result.lowDepth.data)[index] * scaleFeaturesDepth, 100.0f);
      assert(feature(5, index) >= 0);
      assert(feature(5, index) <= 100);
    }
  crf.addPairwiseEnergy(feature, new PottsCompatibility(weightAppearance), DIAG_KERNEL, NORMALIZE_SYMMETRIC);  // addPairwiseBilateral

  // Run, very similar to crf.inference(crfIterations)
  Eigen::MatrixXf crfResult, tmp1, tmp2;

  assert(unary.cols() > 0 && unary.rows() > 0);
  for (int i = 0; i < unary.cols(); ++i)
    for (int j = 0; j < unary.rows(); ++j)
      if (unary(j, i) <= 1e-5) unary(j, i) = 1e-5;

  DenseCRF::expAndNormalize(crfResult, -unary);

  for (unsigned it = 0; it < crfIterations; it++) {
    tmp1 = -unary;
    for (unsigned int k = 0; k < crf.countPotentials(); k++) {
      crf.getPotential(k)->apply(tmp2, crfResult);
      tmp1 -= tmp2;
    }
    DenseCRF::expAndNormalize(crfResult, tmp1);
  }

  // Write segmentation (label with highest prob, map) to image
  // VectorXs map(lowTotal);
  cv::Mat map(lowHeight, lowWidth, CV_8UC1);
  int m;
  for (unsigned i = 0; i < lowTotal; i++) {
    crfResult.col(i).maxCoeff(&m);
    map.data[i] = result.modelData[m].id;
  }

  TOCK("CRF-FULL");

  std::vector<ComponentData> ccStats;
  cv::Mat connectedComponents = connectedLabels(map, &ccStats);
  // cv::imshow("Components", mapLabelToColorImage(connectedComponents));

  // Find mapping from labels to components
  std::map<int, std::list<int>> labelToComponents = mapLabelsToComponents(ccStats);

  const bool onlyKeepLargest = true;
  const bool checkNewModelSize = true;

  // Enforce connectivity (find regions of same label, only keep largest)
  // TODO: This is not always the best decision. Get smarter!
  if (onlyKeepLargest) {
    // skip background label
    for (auto l = std::next(labelToComponents.begin()); l != labelToComponents.end(); l++) {
      // for(auto& l : labelToComponents){
      std::list<int>& labelComponents = l->second;

      // Remove every component, except largest
      std::list<int>::iterator it = labelComponents.begin();
      std::list<int>::iterator s, it2;

      while ((it2 = std::next(it)) != labelComponents.end()) {
        if (ccStats[*it].size < ccStats[*it2].size) {
          s = it;
          it = it2;
        } else {
          s = it2;
        }
        ccStats[*s].label = 255;  // Remove all smaller components (here highlight instead of BG)
        labelComponents.erase(s);
      }
    }
  }

  // TODO Prevent "normal" labels from splitting up
  // Suppress new labels that are too big / too small
  if (allowNew && checkNewModelSize) {
    const int minSize = lowTotal * minRelSizeNew;
    const int maxSize = lowTotal * maxRelSizeNew;
    std::list<int>& l = labelToComponents[nextModelID];

    for (auto& cIndex : l) {
      int size = ccStats[cIndex].size;
      if (size < minSize || size > maxSize) ccStats[cIndex].label = 255;
    }
  }

  // Compute bounding box for each model
  for (SegmentationResult::ModelData& m : result.modelData) {
    for (const int& compIndex : labelToComponents[m.id]) {
      ComponentData& stats = ccStats[compIndex];
      if (stats.left < m.boundingBox.left) m.boundingBox.left = stats.left;
      if (stats.top < m.boundingBox.top) m.boundingBox.top = stats.top;
      if (stats.right > m.boundingBox.right) m.boundingBox.right = stats.right;
      if (stats.bottom > m.boundingBox.bottom) m.boundingBox.bottom = stats.bottom;
    }
    cv::Point2i p = slic.mapToHigh(m.boundingBox.left, m.boundingBox.top);
    m.boundingBox.left = p.x;
    m.boundingBox.top = p.y;
    p = slic.mapToHigh(m.boundingBox.right, m.boundingBox.bottom);
    m.boundingBox.right = p.x;
    m.boundingBox.bottom = p.y;
  }

  // Remove labels that are too close to border
  const int borderSize = 20;
  for (SegmentationResult::ModelData& m : result.modelData) {
    if (m.id == 0) continue;

    if ((m.boundingBox.top < borderSize && m.boundingBox.bottom < borderSize) || (m.boundingBox.left < borderSize && m.boundingBox.right < borderSize) ||
        (m.boundingBox.top > fullHeight - borderSize && m.boundingBox.bottom > fullHeight - borderSize) ||
        (m.boundingBox.left > fullWidth - borderSize && m.boundingBox.right > fullWidth - borderSize)) {
      // TODO This is used more often. Create lambda!
      std::list<int>& l = labelToComponents[m.id];
      for (auto& cIndex : l) {
        ccStats[cIndex].label = 255;
      }
    }
  }

  // Update result (map)
  int* pComponents = (int*)connectedComponents.data;
  //#pragma omp parallel for
  for (unsigned i = 0; i < lowTotal; i++) map.data[i] = ccStats[pComponents[i]].label;

  if (true) {
    std::vector<float> sumsDepth(result.modelData.size(), 0);
    std::vector<float> sumsDeviation(result.modelData.size(), 0);
    std::vector<unsigned> cnts(result.modelData.size(), 0);
    for (unsigned i = 0; i < lowTotal; i++) {
      if (map.data[i] == 255) continue;
      const size_t index = modelIdToIndex[map.data[i]];
      const float d = getDepth(i);
      assert(d >= 0);
      if (!(index < sumsDepth.size())) {
        std::cout << "\n\nindex: " << index << " result.modelData.size() " << result.modelData.size()
                  << " map.data[i]: " << (int)map.data[i] << std::endl
                  << std::flush;
      }
      assert(index < sumsDepth.size());
      sumsDepth[index] += d;
      cnts[index]++;
    }
    for (size_t index = 0; index < result.modelData.size(); ++index)
      result.modelData[index].depthMean = cnts[index] ? sumsDepth[index] / cnts[index] : 0;

    for (unsigned i = 0; i < lowTotal; i++) {
      if (map.data[i] == 255) continue;
      const size_t index = modelIdToIndex[map.data[i]];
      const float d = getDepth(i);
      sumsDeviation[index] += std::abs(result.modelData[index].depthMean - d);
    }
    for (size_t index = 0; index < result.modelData.size(); ++index)
      result.modelData[index].depthStd = cnts[index] ? sumsDeviation[index] / cnts[index] : 0;

    for (unsigned i = 0; i < lowTotal; i++) {
      if (map.data[i] == 255) continue;
      const size_t index = modelIdToIndex[map.data[i]];
      if (index != 0) {
        const float d = getDepth(i);
        if (d > 1.1 * result.modelData[index].depthStd + result.modelData[index].depthMean) {
          // To update mean and std
          sumsDepth[index] -= d;
          sumsDeviation[index] -=
              (std::abs(result.modelData[index].depthMean - d));  // Todo this is only approximating the std, should be good enough
          cnts[index]--;
        }
      }
    }

    // Update mean and std
    for (size_t i = 0; i < result.modelData.size(); ++i) {
      // Careful, see todo comment above
      result.modelData[i].depthMean = cnts[i] ? sumsDepth[i] / cnts[i] : 0;
      result.modelData[i].depthStd = cnts[i] ? sumsDeviation[i] / cnts[i] : 0;
    }
  }

  // Count super-pixel per model (Check whether new model made it)
  for (unsigned k = 0; k < lowTotal; k++) {
    if (map.data[k] == 255) continue;
    result.modelData[modelIdToIndex[map.data[k]]].superPixelCount++;
  }

  // Suppress tiny labels
  // for(unsigned k = 0; k < lowTotal; k++){
  //    auto& l = map[k];
  //    if(result.labelSuperPixelCounts[l] < 5){
  //        result.labelSuperPixelCounts[l] = 0;
  //        if(k > 0) l = map[k-1];
  //        else l = 0;
  //    }
  //}

  if (allowNew) {
    if (result.modelData.back().superPixelCount > 0) {
      // result.numLabels++;
      result.hasNewLabel = true;
    } else {
      result.modelData.pop_back();
    }
  }

  // Upscale result and compute bounding boxes
  result.fullSegmentation = slic.upsample<unsigned char>(map);

  // Check which models are empty
  for (auto& m : result.modelData)
      m.isEmpty = m.superPixelCount <= 0;

#ifdef SHOW_DEBUG_VISUALISATION  // Visualise unary potentials
  const bool writeOverlay = false;
  const bool writeUnaries = false;
  const bool writeICP = false;
  const bool writeSLIC = false;

  const std::string outputPath = "/tmp/maskfusion";
  const int minWrite = 2;

  cv::Mat inputOverlay = showInputOverlay(frame.rgb, result.fullSegmentation);
  if (writeOverlay) cv::imwrite(outputPath + "overlay" + std::to_string(CFRAME) + ".png", inputOverlay);
  if (writeSLIC) cv::imwrite(outputPath + "superpixel" + std::to_string(CFRAME) + ".png", slic.drawSurfelBorders(true));

  unsigned i = 0;
  std::vector<cv::Mat> potentialViews;
  for (; i < numLabels; ++i) {
    cv::Mat imgVisLabel = mapCRFToImage(crfResult, i);
    cv::Mat unaryUpsampling = slic.upsample<cv::Vec3b>(labelDebugImages[i].crfImage);
    cv::Mat icpUpsampling = slic.upsample<cv::Vec3b>(floatToUC3(labelDebugImages[i].icpLow));
    cv::Mat potentialsBefore = slic.drawSurfelBorders(unaryUpsampling, false);
    cv::Mat potentialsAfter = slic.drawSurfelBorders(slic.upsample<cv::Vec3b>(imgVisLabel), false);
    potentialViews.push_back(stackImagesHorizontally({icpUpsampling, potentialsBefore, potentialsAfter}));
    if (writeUnaries) cv::imwrite(outputPath + "unaries" + std::to_string(i) + "-" + std::to_string(CFRAME) + ".png", potentialsBefore);
  }

  // Merge imgPotentialsBeforeAfter images:
  cv::Mat potentialView = stackImagesVertically(potentialViews);
  if (potentialView.rows > 940) {
    float s = 940.0f / potentialView.rows;
    cv::resize(potentialView, potentialView, cv::Size(potentialView.cols * s, potentialView.rows * s));
  }
  imshow("Potentials (before, after)", potentialView);

  if (writeUnaries)
    for (; i < minWrite; ++i)
      cv::imwrite(outputPath + "unaries" + std::to_string(i) + "-" + std::to_string(CFRAME) + ".png",
                  slic.drawSurfelBorders(cv::Mat::zeros(frame.rgb.rows, frame.rgb.cols, CV_8UC3), false));

  i = 0;
  for (; i < result.modelData.size(); i++) {
    SegmentationResult::ModelData& m = result.modelData[i];
    cv::Mat icp;
    m.lowICP.convertTo(icp, CV_8UC3, 255.0);
    if (writeICP) cv::imwrite(outputPath + "ICP" + std::to_string(i) + "-" + std::to_string(CFRAME) + ".png", icp);
  }
  for (; i < minWrite; ++i) {
    if (writeICP)
      cv::imwrite(outputPath + "ICP" + std::to_string(i) + "-" + std::to_string(CFRAME) + ".png",
                  cv::Mat::zeros(frame.rgb.rows, frame.rgb.cols, CV_8UC3));
  }

  cv::waitKey(1);
#endif

  return result;
}
