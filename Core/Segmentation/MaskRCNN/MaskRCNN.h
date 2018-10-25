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

#include <opencv2/imgproc/imgproc.hpp>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"

#include <atomic>
#include <thread>
#include <queue>

#include "../../FrameData.h"

class MaskRCNN {
public:

    struct Result {
        cv::Mat segmentation;
    };

  MaskRCNN(std::queue<FrameDataPointer>* queue = NULL);
  virtual ~MaskRCNN();

  void initialise();
  void* loadModule();

  // Warning, getPyObject requiers a decref:
  inline PyObject* getPyObject(const char* name);

  cv::Mat extractImage();
  void extractClassIDs(std::vector<int>* result);
  void extractBoundingBoxes(std::vector<cv::Rect>* result);
  void executeSequential(FrameDataPointer frameData);
  void startThreadLoop();
  void loop();
  PyObject* createArguments(cv::Mat rgbImage);


private:
    PyObject *pModule;
    PyObject *pExecute;

    // For parallel execution
    std::thread thread;
    std::exception_ptr threadException;
    std::queue<FrameDataPointer>* pQueue = NULL;
};
