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
#ifdef WITH_FREENECT2
#include <iostream>
#include "FreenectLiveReader.h"

#include <opencv2/imgproc/imgproc.hpp>

FreenectLiveReader::FreenectLiveReader(bool fullResolution) :
    LogReader("", false),
//    lastFrameTime(-1),
//    lastGot(-1),
    frameListener(libfreenect2::Frame::Color | libfreenect2::Frame::Depth),
    undistorted_depth(512, 424, 4),
    registered_color(512, 424, 4),
    deviceGood(false){

    std::cout << "Initialising freenect-version: " << LIBFREENECT2_VERSION << std::endl;

#ifndef NDEBUG
    libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Debug));
#endif

#ifdef LIBFREENECT2_WITH_CUDA_SUPPORT
    pipeline = new libfreenect2::CudaPacketPipeline();
    std::cout << "Using freenect with CUDA-pipeline." << std::endl;
#else
    pipeline = new libfreenect2::CpuPacketPipeline();
    std::cout << "Using freenect with CPU-pipeline." << std::endl;
#endif

    if(freenect2.enumerateDevices() == 0)
        std::cerr << "No Freenect device." << std::endl;

    deviceSerial = freenect2.getDefaultDeviceSerialNumber();
    std::cout << "Using freenect device with serial: " << deviceSerial << std::endl;

    if(pipeline) dev = freenect2.openDevice(deviceSerial, pipeline);
    else dev = freenect2.openDevice(deviceSerial);

    dev->setColorFrameListener(&frameListener);
    dev->setIrAndDepthFrameListener(&frameListener);

    deviceGood = dev->start();

    if(deviceGood){
        libfreenect2::Freenect2Device::ColorCameraParams cintr = dev->getColorCameraParams();
        std::cout << "Started KinectV2..."
                  << "\n device serial: " << dev->getSerialNumber()
                  << "\n device firmware: " << dev->getFirmwareVersion()
                  << "\n default color intrinsics: [fx: " << cintr.fx << " fy: " << cintr.fy << "] [cx: " << cintr.cx << " cy: " << cintr.cy << "]" << std::endl;
    }
    freenectRegistration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());

    bufferingLoopActive = true;
    bufferingThread = std::thread(&FreenectLiveReader::bufferLoop, this);
}

FreenectLiveReader::~FreenectLiveReader() {
    bufferingLoopActive = false;
    dev->stop();
    dev->close();
    delete freenectRegistration;
}

FrameDataPointer FreenectLiveReader::processNewest() {

    if (!frameListener.waitForNewFrame(frames, 10000)) {
        std::cerr << "CAMERA TIMEOUT!" << std::endl;
        return NULL;
    }

    FrameDataPointer data = std::make_shared<FrameData>();
    data->allocateRGBD(Resolution::getInstance().width(), Resolution::getInstance().height());

    libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color]; // (1920x1080 BGRX)
    libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth]; // (512x424 float)
    freenectRegistration->apply(rgb, depth, &undistorted_depth, &registered_color);

    // Copy RGB
    size_t s = 0;
    for(size_t y = 0; y < 424; y++){
        for(size_t x = 0; x < 512; x++){
            size_t j0 = 3*(s+x);
            size_t j1 = 4*(s+511-x);
            data->rgb.data[j0] = registered_color.data[j1+2];
            data->rgb.data[j0+1] = registered_color.data[j1+1];
            data->rgb.data[j0+2] = registered_color.data[j1];
        }
        s += 512;
    }

    // Copy Depth
    s = 0;
    for(size_t y = 0; y < 424; y++){
        for(size_t x = 0; x < 512; x++){
            ((float*)data->depth.data)[s+x] = 0.001 * ((float*)undistorted_depth.data)[s+511-x];
        }
        s += 512;
    }

    frameListener.release(frames);
    //if (flipColors) data->flipColors();

    return data;
}

//const std::string FreenectLiveReader::getFile() { return Parse::get().baseDir().append("live"); }

FrameDataPointer FreenectLiveReader::getFrameData() { return atomic_load(&currentDataPointer); }

int FreenectLiveReader::getNumFrames() { return std::numeric_limits<int>::max(); }

bool FreenectLiveReader::hasMore() { return true; }

void FreenectLiveReader::bufferLoop(){
    std::cout << "Started Kinect-buffering thread with id: " << std::this_thread::get_id() << std::endl;
    while (bufferingLoopActive) {
        // currentDataPointer = processNewest(); c++20
        atomic_store(&currentDataPointer, processNewest());
    }
}
#endif
