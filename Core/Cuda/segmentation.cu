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

#include "cudafuncs.cuh"
#include "convenience.cuh"
#include "operators.cuh"
#include "segmentation.cuh"

__global__ void bilateralFilter_Kernel(int w,
                                       int h,
                                       int radius,
                                       int minValues,
                                       float sigmaDepth,
                                       float sigmaColor,
                                       float sigmaLocation,
                                       const PtrStepSz<uchar4> inputRGBA,
                                       const PtrStepSz<float> inputDepth,
                                       PtrStepSz<float> output){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= h || x >= w) return;

    //output.ptr(y)[x] = inputRGBA.ptr(y)[x].w / 255.0f;//inputDepth.ptr(y)[x];


    //const float sigma_space2_inv_half = 0.024691358; // 0.5 / (sigma_space * sigma_space)
    //const float sigma_color2_inv_half = 555.556; // 0.5 / (sigma_color * sigma_color)
    const float i_sigma_depth_2 = 0.5f / (sigmaDepth*sigmaDepth);
    const float i_sigma_color_2 = 0.5f / (sigmaColor*sigmaColor);
    const float i_sigma_location_2 = 0.5f / (sigmaLocation*sigmaLocation);

    const int x1 = max(x-radius,0);
    const int y1 = max(y-radius,0);
    const int x2 = min(x+radius, w-1);
    const int y2 = min(y+radius, h-1);

    float weight, location_diff, color_diff, depth_diff;
    float sum_v = 0;
    float sum_w = 0;

    for(int cy = y1; cy <= y2; ++cy){
        for(int cx = x1; cx <= x2; ++cx){

            location_diff = (x - cx) * (x - cx) + (y - cy) * (y - cy);

            color_diff = (inputRGBA.ptr(y)[x].x - inputRGBA.ptr(cy)[cx].x) * (inputRGBA.ptr(y)[x].x - inputRGBA.ptr(cy)[cx].x) +
                    (inputRGBA.ptr(y)[x].y - inputRGBA.ptr(cy)[cx].y) * (inputRGBA.ptr(y)[x].y - inputRGBA.ptr(cy)[cx].y) +
                    (inputRGBA.ptr(y)[x].z - inputRGBA.ptr(cy)[cx].z) * (inputRGBA.ptr(y)[x].z - inputRGBA.ptr(cy)[cx].z);

            depth_diff = (inputDepth.ptr(y)[x] - inputDepth.ptr(cy)[cx]);
            depth_diff *= depth_diff;

            weight = exp(-location_diff*i_sigma_location_2 -depth_diff*i_sigma_depth_2 -color_diff*i_sigma_color_2);
            sum_v += weight * inputDepth.ptr(cy)[cx];
            sum_w += weight;
        }
    }

    // TODO if min values
    output.ptr(y)[x] = sum_v / sum_w;

}

void bilateralFilter(const DeviceArray2D<uchar4> inputRGB,
                     const DeviceArray2D<float> inputDepth,
                     const DeviceArray2D<float> outputDepth,
                     int radius, int minValues, float sigmaDepth, float sigmaColor, float sigmaLocation){

    const int w = inputDepth.cols();
    const int h = inputDepth.rows();
    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = getGridDim (w, block.x);
    grid.y = getGridDim (h, block.y);
    bilateralFilter_Kernel<<<grid, block>>>(w, h, radius, minValues, sigmaDepth, sigmaColor, sigmaLocation, inputRGB, inputDepth, outputDepth);

    cudaCheckError();
    cudaSafeCall (cudaDeviceSynchronize ());

}

// Generate a vertex map 'vmap' based on the depth map 'depth' and camera parameters

__device__ __forceinline__ float3 getFloat3(int w, int h, const PtrStepSz<float> img32FC3, int x, int y)
{
    return { img32FC3.ptr(y)[x],
                img32FC3.ptr(y+h)[x],
                img32FC3.ptr(y+2*h)[x] };
}

__device__ float getConcavityTerm(int w, int h, const PtrStepSz<float> vmap, const PtrStepSz<float> nmap, const float3& v, const float3& n, int x_n, int y_n) {
    const float3 v_n = getFloat3(w, h, vmap, x_n, y_n);
    const float3 n_n = getFloat3(w, h, nmap, x_n, y_n);
    if(dot(v_n-v,n) < 0) return 0;
    return 1-dot(n_n,n);
    //return acos(dot(n_n,n));
}


__device__ float getDistanceTerm(int w, int h, const PtrStepSz<float> vmap, const float3& v, const float3& n, int x_n, int y_n) {
    const float3 v_n = getFloat3(w, h, vmap, x_n, y_n);
    float3 d = v_n - v;
    return fabs(dot(d, n));
}

//__global__ void computeGeometricSegmentation_Kernel(int w, int h, const PtrStepSz<float> vmap, const PtrStepSz<float> nmap, cudaSurfaceObject_t output, float threshold)
//__global__ void computeGeometricSegmentation_Kernel(int w, int h, const PtrStepSz<float> vmap, const PtrStepSz<float> nmap, PtrStepSz<unsigned char> output, float threshold){
__global__ void computeGeometricSegmentation_Kernel(int w, int h,
                                                    const PtrStepSz<float> vmap,
                                                    const PtrStepSz<float> nmap,
                                                    PtrStepSz<float> output,
                                                    float wD, float wC){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= h || x >= w) return;

    const int radius = 1;
    if (x < radius || x >= w-radius || y < radius || y >= h-radius){
        //surf2Dwrite(1.0f, output, x_out, y);
        output.ptr(y)[x] = 1.0f;
        //output.ptr(y)[x] = 255;
        return;
    }

    //TODO handle special case: missing depth!

    const float3 v = getFloat3(w, h, vmap, x, y);
    const float3 n = getFloat3(w, h, nmap, x, y);
    if(v.z <= 0.0f) {
        output.ptr(y)[x] = 1.0f;
        return;
    }

    float c = 0.0f;
    c = fmax(getConcavityTerm(w, h, vmap, nmap, v, n, x-radius, y-radius), c);
    c = fmax(getConcavityTerm(w, h, vmap, nmap, v, n, x, y-radius), c);
    c = fmax(getConcavityTerm(w, h, vmap, nmap, v, n, x+radius, y-radius), c);
    c = fmax(getConcavityTerm(w, h, vmap, nmap, v, n, x-radius, y), c);
    c = fmax(getConcavityTerm(w, h, vmap, nmap, v, n, x+radius, y), c);
    c = fmax(getConcavityTerm(w, h, vmap, nmap, v, n, x-radius, y+radius), c);
    c = fmax(getConcavityTerm(w, h, vmap, nmap, v, n, x, y+radius), c);
    c = fmax(getConcavityTerm(w, h, vmap, nmap, v, n, x+radius, y+radius), c);
    c = fmax(c,0.0f);
    c *= wC;
    //    if(c < 0.99) c = 0;
    //    else c = 1;

    float d = 0.0f;
    d = fmax(getDistanceTerm(w, h, vmap, v, n, x-radius, y-radius), d);
    d = fmax(getDistanceTerm(w, h, vmap, v, n, x, y-radius), d);
    d = fmax(getDistanceTerm(w, h, vmap, v, n, x+radius, y-radius), d);
    d = fmax(getDistanceTerm(w, h, vmap, v, n, x-radius, y), d);
    d = fmax(getDistanceTerm(w, h, vmap, v, n, x+radius, y), d);
    d = fmax(getDistanceTerm(w, h, vmap, v, n, x-radius, y+radius), d);
    d = fmax(getDistanceTerm(w, h, vmap, v, n, x, y+radius), d);
    d = fmax(getDistanceTerm(w, h, vmap, v, n, x+radius, y+radius), d);
    d *= wD;

    float edgeness = max(c,d);

    //surf2Dwrite(edgeness, output, x_out, y);
    output.ptr(y)[x] = fmin(1.0f, edgeness);
}

__global__ void f_erode_Kernel(int w, int h, const PtrStepSz<float> intput, PtrStepSz<float> output) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= h || x >= w) return;

    float r = intput.ptr(y)[x];
    if (x < 1 || x >= w-1 || y < 1 || y >= h-1) output.ptr(y)[x] = r;

    r = fmin(intput.ptr(y-1)[x-1], r);
    r = fmin(intput.ptr(y-1)[x], r);
    r = fmin(intput.ptr(y-1)[x+1], r);
    r = fmin(intput.ptr(y)[x-1], r);
    r = fmin(intput.ptr(y)[x+1], r);
    r = fmin(intput.ptr(y+1)[x-1], r);
    r = fmin(intput.ptr(y+1)[x], r);
    r = fmin(intput.ptr(y+1)[x+1], r);
    output.ptr(y)[x] = r;
}

__global__ void f_dilate_Kernel(int w, int h, const PtrStepSz<float> intput, PtrStepSz<float> output) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= h || x >= w) return;

    float r = intput.ptr(y)[x];
    if (x < 1 || x >= w-1 || y < 1 || y >= h-1) output.ptr(y)[x] = r;

    r = fmax(intput.ptr(y-1)[x-1], r);
    r = fmax(intput.ptr(y-1)[x], r);
    r = fmax(intput.ptr(y-1)[x+1], r);
    r = fmax(intput.ptr(y)[x-1], r);
    r = fmax(intput.ptr(y)[x+1], r);
    r = fmax(intput.ptr(y+1)[x-1], r);
    r = fmax(intput.ptr(y+1)[x], r);
    r = fmax(intput.ptr(y+1)[x+1], r);
    output.ptr(y)[x] = r;
}

__global__ void erode_Kernel(int w, int h, int radius, const PtrStepSz<unsigned char> intput, PtrStepSz<unsigned char> output) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= h || x >= w) return;
    const int x1 = max(x-radius,0);
    const int y1 = max(y-radius,0);
    const int x2 = min(x+radius, w-1);
    const int y2 = min(y+radius, h-1);
    output.ptr(y)[x] = 255;
    for (int cy = y1; cy <= y2; ++cy){
        for (int cx = x1; cx <= x2; ++cx){
            if (cy == y && cx == x) continue;
            if (intput.ptr(cy)[cx] == 0) {
                output.ptr(y)[x] = 0;
                return;
            }
        }
    }
}

__global__ void dilate_Kernel(int w, int h, int radius, const PtrStepSz<unsigned char> intput, PtrStepSz<unsigned char> output) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= h || x >= w) return;
    const int x1 = max(x-radius,0);
    const int y1 = max(y-radius,0);
    const int x2 = min(x+radius, w-1);
    const int y2 = min(y+radius, h-1);
    output.ptr(y)[x] = 0;
    for (int cy = y1; cy <= y2; ++cy){
        for (int cx = x1; cx <= x2; ++cx){
            if (cy == y && cx == x) continue;
            if (intput.ptr(cy)[cx] == 255) {
                output.ptr(y)[x] = 255;
                return;
            }
        }
    }
}

__global__ void threshold_Kernel(const PtrStepSz<float> input, PtrStepSz<unsigned char> output, float threshold) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= input.rows || x >= input.cols) return;
    output.ptr(y)[x] = input.ptr(y)[x] > threshold ? 255 : 0;
}

__global__ void invert_Kernel(const PtrStepSz<unsigned char> input, PtrStepSz<unsigned char> output) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= input.rows || x >= input.cols) return;
    output.ptr(y)[x] = 255 - input.ptr(y)[x];
}

//__global__ void morphGeometricSegmentation_Kernel(int w, int h, const PtrStepSz<float> input, const PtrStepSz<float> output)
//{

//}


void computeGeometricSegmentationMap(const DeviceArray2D<float> vmap,
                                     const DeviceArray2D<float> nmap,
                                     const DeviceArray2D<float> output,
                                     float wD, float wC){
    const int w = vmap.cols();
    const int h = vmap.rows() / 3;
    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = getGridDim (w, block.x);
    grid.y = getGridDim (h, block.y);
    //std::cout << "Running block, info: " << grid.x << " " << grid.y << " - wh: " << w << " " << h << std::endl;
    computeGeometricSegmentation_Kernel<<<grid, block>>>(w, h, vmap, nmap, output, wD, wC);

    cudaCheckError();
    cudaSafeCall (cudaDeviceSynchronize ());
}

void thresholdMap(const DeviceArray2D<float> input,
                  const DeviceArray2D<unsigned char> output,
                  float threshold){
    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = getGridDim (input.cols(), block.x);
    grid.y = getGridDim (input.rows(), block.y);
    threshold_Kernel<<<grid, block>>>(input, output, threshold);
}

void invertMap(const DeviceArray2D<unsigned char> input,
               const DeviceArray2D<unsigned char> output){
    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = getGridDim (input.cols(), block.x);
    grid.y = getGridDim (input.rows(), block.y);
    invert_Kernel<<<grid, block>>>(input, output);
}


void morphGeometricSegmentationMap(const DeviceArray2D<float> data,
                                   const DeviceArray2D<float> buffer){

    const int w = data.cols();
    const int h = data.rows();
    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = getGridDim (w, block.x);
    grid.y = getGridDim (h, block.y);
    f_dilate_Kernel<<<grid, block>>>(w, h, data, buffer);
    f_erode_Kernel<<<grid, block>>>(w, h, buffer, data);
    f_dilate_Kernel<<<grid, block>>>(w, h, data, buffer);
    f_erode_Kernel<<<grid, block>>>(w, h, buffer, data);
    f_dilate_Kernel<<<grid, block>>>(w, h, data, buffer);
    f_erode_Kernel<<<grid, block>>>(w, h, buffer, data);

    cudaCheckError();
    cudaSafeCall (cudaDeviceSynchronize ());
}

void morphGeometricSegmentationMap(const DeviceArray2D<unsigned char> data,
                                const DeviceArray2D<unsigned char> buffer,
                                int radius,
                                int iterations){

    const int w = data.cols();
    const int h = data.rows();
    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = getGridDim (w, block.x);
    grid.y = getGridDim (h, block.y);
    for (int i = 0; i < iterations; ++i) {
        dilate_Kernel<<<grid, block>>>(w, h, radius, data, buffer);
         cudaSafeCall (cudaDeviceSynchronize ());
        erode_Kernel<<<grid, block>>>(w, h, radius, buffer, data);
         cudaSafeCall (cudaDeviceSynchronize ());
        //erode_Kernel<<<grid, block>>>(w, h, radius, buffer, data);
    }
    cudaCheckError();
    cudaSafeCall (cudaDeviceSynchronize ());
}



//__global__ void computeGeometricSegmentation_Kernel(const PtrStepSz<float> vmap, const PtrStepSz<float> nmap, PtrStepSz<float> output, float threshold)
//{
//    int x = threadIdx.x + blockIdx.x * blockDim.x;
//    int y = threadIdx.y + blockIdx.y * blockDim.y;

//    if (x < 1 || x >= output.cols-1 || y < 1 || y >= output.rows-1){
//        output.ptr(y)[x] = 1;
//        return;
//    }

//    float c = 1;
//    c = min(getConvexityTerm(vmap, nmap, x, x-1, y, y-1), c);

//    output.ptr(y)[x] = c;
//}

//void computeGeometricSegmentationMap(const DeviceArray2D<float> vmap,
//                                     const DeviceArray2D<float> nmap,
//                                     DeviceArray2D<float> segmentationMap,
//                                     float threshold){
//    dim3 block (32, 8);
//    dim3 grid (1, 1, 1);
//    grid.x = getGridDim (segmentationMap.cols (), block.x);
//    grid.y = getGridDim (segmentationMap.rows (), block.y);
//    computeGeometricSegmentation_Kernel<<<grid, block>>>(vmap, nmap, segmentationMap, threshold);

//    //cudaSafeCall (cudaGetLastError ());

//    cudaCheckError();
//    cudaSafeCall (cudaDeviceSynchronize ());
//}
