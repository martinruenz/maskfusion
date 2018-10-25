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
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#ifndef CUDA_CONVENIENCE_CUH_
#define CUDA_CONVENIENCE_CUH_

#include <cuda_runtime_api.h>
#include <cstdlib>
#include <iostream>
#include <stdio.h>

#define STR2(x) #x
#define STRINGIFY(x) STR2(x)

#define FILE_LINE __FILE__ ":" STRINGIFY(__LINE__)

static inline int getGridDim(int x, int y)
{
    return (x + y - 1) / y;
}

/*static inline void cudaCheckError(std::string fileline){
    cudaSafeCall(cudaGetLastError(), fileline);
}*/

#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define cudaCheckError() __cudaSafeCall(cudaGetLastError(), __FILE__, __LINE__)
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
                printf("%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
                file, line, cudaGetErrorString(err) );
        exit(-1);
    }
}

static inline void cudaPrintMemory(){
    size_t free, total;
    cudaMemGetInfo(&free,&total);
    std::cout << "Cuda memory: " << float(free)/(1024.0*1024.0) << " / " << float(total)/(1024.0*1024.0) << "MB" << std::endl;
}

#endif /* CUDA_CONVENIENCE_CUH_ */
