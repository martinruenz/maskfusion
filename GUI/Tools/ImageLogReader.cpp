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

#include "ImageLogReader.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/algorithm.hpp>
#include <boost/algorithm/string.hpp>
#include <stdexcept>
#include <iomanip>
#include <fstream>

static std::string cvTypeToString(int type) {
  std::string r;
  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);
  switch (depth) {
    case CV_8U:
      r = "8U";
      break;
    case CV_8S:
      r = "8S";
      break;
    case CV_16U:
      r = "16U";
      break;
    case CV_16S:
      r = "16S";
      break;
    case CV_32S:
      r = "32S";
      break;
    case CV_32F:
      r = "32F";
      break;
    case CV_64F:
      r = "64F";
      break;
    default:
      r = "User";
      break;
  }
  r += "C";
  r += (chans + '0');
  return r;
}

ImageLogReader::ImageLogReader(std::string colorDirectory, std::string depthDirectory, std::string maskDirectory, unsigned indexWidth,
                               std::string colorPrefix, std::string depthPrefix, std::string maskPrefix, bool flipColors)
    : LogReader(colorDirectory, flipColors),
      depthImagesDir(depthDirectory),
      maskImagesDir(maskDirectory),
      colorPre(colorPrefix),
      depthPre(depthPrefix),
      maskPre(maskPrefix),
      indexW(indexWidth) {
  using namespace boost::filesystem;
  using namespace boost::algorithm;

  const std::vector<std::string> rgbExtensions = {".jpg", ".png", ".ppm"};
  const std::vector<std::string> depthExtensions = {".exr", ".png"};
  const std::vector<std::string> maskExtensions = {".png", ".pgm"};

  // Overlapping directories but no distinct prefixes? Try default prefixes..
  if (((depthDirectory == colorDirectory) || (maskDirectory == colorDirectory) || (maskDirectory == depthDirectory)) &&
      ((depthPre == colorPre) && (maskPre == colorPre) && (maskPre == depthPre))) {
    colorPre = "Color";
    depthPre = "Depth";
    maskPre = "Mask";
  }

  auto countFilesInDir = [](const std::string& path, const std::string& prefix, const std::vector<std::string>& extension,
                            std::string& outExt) -> unsigned {
    unsigned result = 0;
    outExt = "";
    for (auto it = directory_iterator(path); it != directory_iterator(); ++it) {
      if (is_regular_file(it->status())) {
        std::string ext = it->path().extension().string();
        std::string name = it->path().stem().string();
        to_lower(ext);

        if (name.substr(0, prefix.length()) == prefix) {
          for (const std::string& e : extension) {
            if (e == ext) {
              if (outExt == "")
                outExt = ext;
              else if (outExt != ext)
                throw std::invalid_argument("Error: Files in the dataset ( " + path + ", " + prefix +
                                            ") are required to have the same extension.");
              result++;
              break;
            }
          }
        }
      }
    }
    return result;
  };

  unsigned numColorImages = countFilesInDir(colorDirectory, colorPre, rgbExtensions, colorExt);
  unsigned numDepthImages = countFilesInDir(depthDirectory, depthPre, depthExtensions, depthExt);
  unsigned numMaskImages = countFilesInDir(maskDirectory, maskPre, maskExtensions, maskExt);

  if (numMaskImages > 0) {
    hasMasksGT = true;
    maxMasks = numMaskImages;
  }

  if (numColorImages != numDepthImages) throw std::invalid_argument("Error: Number of RGB-frames != Depth-frames!");
  if (hasMasksGT && (numColorImages != numMaskImages)) throw std::invalid_argument("Error: Number of RGB-frames != Mask-frames!");

  numFrames = numColorImages;

  // Find start index
  int index = 0;
  for (; index < 2; index++) {
    std::stringstream ss;
    ss << std::setw(indexW) << std::setfill('0') << index;
    std::string path = colorDirectory + colorPre + ss.str() + colorExt;
    if (exists(path)) {
      startIndex = index;
      break;
    }
  }
  if (index == 2) throw std::invalid_argument("Error: Could not find start index.");

  std::cout << "Opened dataset with " << numFrames << " frames, starting with index: " << startIndex;
  if (hasMasksGT) std::cout << " The dataset also provides masks.";
  std::cout << std::endl;

  // Try to find calibration file in color directory
  const std::string calibrationFile = colorDirectory + "/calibration.txt";
  if (exists(calibrationFile)) LogReader::calibrationFile = calibrationFile;

  currentFrame = -1;
  frames.resize(numFrames);
  startBufferLoop();
}

ImageLogReader::~ImageLogReader() { stopBufferLoop(); }

void ImageLogReader::getPrevious() { assert(0); }

void ImageLogReader::getNext() {
  if (bufferingLoopActive)
    bufferingCondition.notify_one();
  else
    assert(0);  // bufferFrames();

  currentFrame += 1;
}

void ImageLogReader::fastForward(int frame) {
  stopBufferLoop();
  //frames.clear();

  currentFrame = frame;
  nextBufferIndex = frame;

  startBufferLoop();
  bufferFrames();
}

void ImageLogReader::bufferFrames() {
  std::cout << "bufferFrames()" << std::endl;
  if (bufferingLoopActive)
    bufferingCondition.notify_one();
  else
    bufferFramesImpl();
}

void ImageLogReader::startBufferLoop() {
  std::cout << "startBufferLoop()" << std::endl;
  bufferingLoopActive = true;
  bufferingThread = std::thread(&ImageLogReader::bufferLoop, this);
}

void ImageLogReader::stopBufferLoop() {
  std::cout << "stopBufferLoop()" << std::endl;
  while (bufferingLoopActive || !bufferingThread.joinable()) {
    bufferingLoopActive = false;
    bufferingCondition.notify_one();
  }
  bufferingThread.join();
}

void ImageLogReader::bufferLoop() {
  std::cout << "Started data-buffering thread with id: " << std::this_thread::get_id() << std::endl;

  std::unique_lock<std::mutex> lock(bufferingMutex);
  while (bufferingLoopActive) {
    bufferFramesImpl();
    if (bufferingLoopActive) bufferingCondition.wait(lock);
  }
}

void ImageLogReader::bufferFramesImpl() {
  if (int(nextBufferIndex) - minBuffered + 1 > int(currentFrame)) return;
  for (unsigned i = 0; i < 15 && nextBufferIndex < frames.size(); ++i, ++nextBufferIndex) {
    frames[nextBufferIndex] = loadFrameFromDrive(nextBufferIndex);
  }
}

FrameDataPointer ImageLogReader::loadFrameFromDrive(const size_t& index) {
  FrameDataPointer result = std::make_shared<FrameData>();

  // Get path to image files
  std::stringstream ss;
  ss << std::setw(indexW) << std::setfill('0') << index + startIndex;
  std::string indexStr = ss.str();

  std::string depthImagePath = depthImagesDir + depthPre + indexStr + depthExt;
  if (!boost::filesystem::exists(depthImagePath)) throw std::invalid_argument("Could not find depth-image file: " + depthImagePath);

  std::string rgbImagePath = file + colorPre + indexStr + colorExt;
  if (!boost::filesystem::exists(rgbImagePath)) throw std::invalid_argument("Could not find rgb-image file: " + rgbImagePath);

  // Load mask ids
  //std::string maskImagePath = maskImagesDir + maskPre + std::to_string(index);
  std::string maskImagePath = maskImagesDir + maskPre + indexStr;
  std::string maskDescrPath = maskImagePath + ".txt";
  maskImagePath += maskExt;
  if (hasMasksGT) {
    if (!boost::filesystem::exists(maskImagePath)) throw std::invalid_argument("Could not find mask-image file: " + maskImagePath);
    if (boost::filesystem::exists(maskDescrPath)) loadMaskIDs(maskDescrPath, &result->classIDs, &result->rois);
  }


  // Load RGB
  result->rgb = cv::imread(rgbImagePath);
  if (result->rgb.total() == 0) throw std::invalid_argument("Could not read rgb-image file.");
  result->flipColors();

  // Load Depth
  result->depth = cv::imread(depthImagePath, cv::IMREAD_UNCHANGED);
  if (result->depth.total() == 0) throw std::invalid_argument("Could not read depth-image file. (Empty)");
  if (result->depth.type() != CV_32FC1) {
    cv::Mat newDepth(result->depth.rows, result->depth.cols, CV_32FC1);
    if (result->depth.type() == CV_32FC3) {
      unsigned depthIdx = 0;
      for (int i = 0; i < result->depth.rows; ++i) {
        cv::Vec3f* pixel = result->depth.ptr<cv::Vec3f>(i);
        for (int j = 0; j < result->depth.cols; ++j) ((float*)newDepth.data)[depthIdx++] = pixel[j][0];
      }

    } else if (result->depth.type() == CV_16UC1) {
      //std::cout << "Warning -- your depth scale is likely to mismatch. Check ImageLogReader.cpp!" << std::endl;
      unsigned depthIdx = 0;
      for (int i = 0; i < result->depth.rows; ++i) {
        unsigned short* pixel = result->depth.ptr<unsigned short>(i);
        for (int j = 0; j < result->depth.cols; ++j) ((float*)newDepth.data)[depthIdx++] = 0.001f * pixel[j];
        //for (int j = 0; j < result->depth.cols; ++j) ((float*)newDepth.data)[depthIdx++] = 0.0002f * pixel[j]; // FIXME
      }
    } else {
      throw std::invalid_argument("Unsupported depth-files: " + cvTypeToString(result->depth.type()));
    }
    result->depth = newDepth;
  }

  // Load Mask
  if (hasMasksGT && (index < maxMasks)) {
    result->mask = cv::imread(maskImagePath, cv::IMREAD_GRAYSCALE);
    if (result->mask.total() != result->rgb.total()) throw std::invalid_argument("Could not read mask-image file.");
    if (!result->mask.isContinuous() || result->mask.type() != CV_8UC1) throw std::invalid_argument("Incompatible mask image.");
  }

  result->timestamp = index * 1000.0f / rateHz;
  result->index = index;

  return result;
}

FrameDataPointer ImageLogReader::getFrameData() {
  // assert(frames[currentFrame] != 0);
  if (currentFrame < 0) return NULL;

  bool bufferFail = false;
  while (!frames[currentFrame] || frames[currentFrame]->depth.total() == 0) {
    usleep(1);
    bufferFail = true;
  }
  if (bufferFail) std::cout << "Buffering failure." << std::endl;
  return frames[currentFrame];
}

void ImageLogReader::loadMaskIDs(const std::string &descrFile, std::vector<int>* outClassIds, std::vector<cv::Rect>* outROIs) const {
    assert(outClassIds->size() == 0 && outROIs->size() == 0);
    std::ifstream file(descrFile);
    std::string item, line;
    outClassIds->push_back(0); // Mask==0 ist always background
    std::getline(file, line);
    std::stringstream ss(line);
    while (getline(ss, item, ' ')) if(item.size() > 0) outClassIds->push_back(stoi(item));

    // get bounding-boxes
    while (std::getline(file, line)){
        std::istringstream iss(line);
        int a, b, c, d;
        if (!(iss >> a >> b >> c >> d))
            throw std::invalid_argument("Extracting bounding-box failed.");
        outROIs->push_back(cv::Rect(b,a,d-b,c-a));
    }
    if(outROIs->size()>0 && outROIs->size()!=outClassIds->size()-1) throw std::invalid_argument("Bounding-boxes provided, but number does not match class ids.");
}

int ImageLogReader::getNumFrames() { return numFrames; }

bool ImageLogReader::hasMore() { return currentFrame + 1 < numFrames; }

bool ImageLogReader::rewind() {
  assert(0 && "rewind");

  return false;
}
