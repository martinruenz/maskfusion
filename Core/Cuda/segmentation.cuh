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



#ifndef CUDA_SEGMENTATION_CUH_
#define CUDA_SEGMENTATION_CUH_

// Normal and verterx maps should already be available, see nmaps_curr_, vmaps_curr_
// Also, already generated, since performTracking was called. (initialises values)
#include "containers/device_array.hpp"
#include "types.cuh"

void computeGeometricSegmentationMap(const DeviceArray2D<float> vmap,
                                     const DeviceArray2D<float> nmap,
                                     const DeviceArray2D<float> output,
                                     //const DeviceArray2D<unsigned char> output,
                                     float wD, float wC);

void thresholdMap(const DeviceArray2D<float> input,
                  const DeviceArray2D<unsigned char> output,
                  float threshold);

void invertMap(const DeviceArray2D<unsigned char> input,
               const DeviceArray2D<unsigned char> output);

void morphGeometricSegmentationMap(const DeviceArray2D<float> data,
                                   const DeviceArray2D<float> buffer);

void morphGeometricSegmentationMap(const DeviceArray2D<unsigned char> data,
                                   const DeviceArray2D<unsigned char> buffer,
                                   int radius,
                                   int iterations);

void bilateralFilter(const DeviceArray2D<uchar4> inputRGB,
                     const DeviceArray2D<float> inputDepth,
                     const DeviceArray2D<float> outputDepth,
                     int radius, int minValues, float sigmaDepth, float sigmaColor, float sigmaLocation);

#endif
