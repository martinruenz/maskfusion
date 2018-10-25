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

#include <stdio.h>
//#include <stdlib.h>
//#include <poll.h>
//#include <signal.h>
#include <atomic>
#include <thread>

//#include <Utils/Parse.h>

#include "LogReader.h"

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/logger.h>

class FreenectLiveReader : public LogReader {
public:
    FreenectLiveReader(bool fullResolution=false);

    virtual ~FreenectLiveReader();

    void getNext() {}
    FrameDataPointer processNewest();

    int getNumFrames();

    bool hasMore();

    bool rewind() { return false; }

    void getPrevious() {}

    void fastForward(int frame) {}

    void bufferLoop();

    const std::string getFile() { return ""; }

    FrameDataPointer getFrameData();

    void setAuto(bool value) {}

    bool isDeviceGood() { return deviceGood; }

private:

    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device *dev = 0;
    libfreenect2::PacketPipeline *pipeline = 0;
    libfreenect2::Registration* freenectRegistration = 0;
    libfreenect2::SyncMultiFrameListener frameListener;
    libfreenect2::FrameMap frames;
    libfreenect2::Frame undistorted_depth, registered_color;
    std::string deviceSerial = "";


    FrameDataPointer currentDataPointer;
    //std::atomic<FrameDataPointer> currentDataPointer; c++20

    bool deviceGood;

    // BufferThread
    //std::mutex bufferingMutex;
    std::atomic<bool> bufferingLoopActive;
    std::thread bufferingThread;
};
