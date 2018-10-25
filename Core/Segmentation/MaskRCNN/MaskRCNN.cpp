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

#include "MaskRCNN.h"
#include "../../FrameData.h"
#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Utils/Stopwatch.h"

MaskRCNN::MaskRCNN(std::queue<FrameDataPointer>* queue){
    pQueue = queue;
    if(pQueue) startThreadLoop();
    else initialise();
}

MaskRCNN::~MaskRCNN() {
    Py_XDECREF(pModule);
    Py_XDECREF(pExecute);
    Py_Finalize();
}

void MaskRCNN::initialise(){

    std::cout << "* Initialising MaskRCNN (thread: " << std::this_thread::get_id() << ") ..." << std::endl;

    Py_SetProgramName((wchar_t*)L"MaskRCNN");
    Py_Initialize();
    wchar_t const * argv2[] = { L"MaskRCNN.py" };
    PySys_SetArgv(1, const_cast<wchar_t**>(argv2));

    // Load module
    loadModule();

    // Get function
    pExecute = PyObject_GetAttrString(pModule, "execute");
    if(pExecute == NULL || !PyCallable_Check(pExecute)) {
        if(PyErr_Occurred()) {
            std::cout << "Python error indicator is set:" << std::endl;
            PyErr_Print();
        }
        throw std::runtime_error("Could not load function 'execute' from MaskRCNN module.");
    }
    std::cout << "* Initialised MaskRCNN" << std::endl;
}

void* MaskRCNN::loadModule(){
    std::cout << " * Loading module..." << std::endl;
    pModule = PyImport_ImportModule("MaskRCNN");
    if(pModule == NULL) {
        if(PyErr_Occurred()) {
            std::cout << "Python error indicator is set:" << std::endl;
            PyErr_Print();
        }
        throw std::runtime_error("Could not open MaskRCNN module.");
    }
    import_array();
    return 0;
}

PyObject *MaskRCNN::getPyObject(const char* name){
    PyObject* obj = PyObject_GetAttrString(pModule, name);
    if(!obj || obj == Py_None) throw std::runtime_error(std::string("Failed to get python object: ") + name);
    return obj;
}

cv::Mat MaskRCNN::extractImage(){
    PyObject* pImage = getPyObject("current_segmentation");
    PyArrayObject *pImageArray = (PyArrayObject*)(pImage);
    //assert(pImageArray->flags & NPY_ARRAY_C_CONTIGUOUS);

    unsigned char* pData = (unsigned char*)PyArray_GETPTR1(pImageArray,0);
    npy_intp h = PyArray_DIM(pImageArray,0);
    npy_intp w = PyArray_DIM(pImageArray,1);

    cv::Mat result;
    cv::Mat(h,w, CV_8UC1, pData).copyTo(result);
    Py_DECREF(pImage);
    return result;
}

void MaskRCNN::extractClassIDs(std::vector<int>* result){
    assert(result->size() == 0);
    PyObject* pClassList = getPyObject("current_class_ids");
    if(!PySequence_Check(pClassList)) throw std::runtime_error("pClassList is not a sequence.");
    Py_ssize_t n = PySequence_Length(pClassList);
    result->reserve(n+1);
    result->push_back(0); // Background
    for (int i = 0; i < n; ++i) {
        PyObject* o = PySequence_GetItem(pClassList, i);
        assert(PyLong_Check(o));
        result->push_back(PyLong_AsLong(o));
        Py_DECREF(o);
    }
    Py_DECREF(pClassList);
}

void MaskRCNN::extractBoundingBoxes(std::vector<cv::Rect> *result){
    assert(result->size() == 0);
    PyObject* pRoiList = getPyObject("current_bounding_boxes");
    if(!PySequence_Check(pRoiList)) throw std::runtime_error("pRoiList is not a sequence.");
    Py_ssize_t n = PySequence_Length(pRoiList);
    result->reserve(n);
    for (int i = 0; i < n; ++i) {
        PyObject* pRoi = PySequence_GetItem(pRoiList, i);
        assert(PySequence_Check(pRoi));
        Py_ssize_t ncoords = PySequence_Length(pRoi);
        assert(ncoords==4);

        PyObject* c0 = PySequence_GetItem(pRoi, 0);
        PyObject* c1 = PySequence_GetItem(pRoi, 1);
        PyObject* c2 = PySequence_GetItem(pRoi, 2);
        PyObject* c3 = PySequence_GetItem(pRoi, 3);
        assert(PyLong_Check(c0) && PyLong_Check(c1) && PyLong_Check(c2) && PyLong_Check(c3));

        int a = PyLong_AsLong(c0);
        int b = PyLong_AsLong(c1);
        int c = PyLong_AsLong(c2);
        int d = PyLong_AsLong(c3);
        Py_DECREF(c0);
        Py_DECREF(c1);
        Py_DECREF(c2);
        Py_DECREF(c3);

        result->push_back(cv::Rect(b,a,d-b,c-a));
        Py_DECREF(pRoi);
    }
    Py_DECREF(pRoiList);
}

void MaskRCNN::executeSequential(FrameDataPointer frameData){
    Py_XDECREF(PyObject_CallFunctionObjArgs(pExecute, createArguments(frameData->rgb), NULL));
    extractClassIDs(&frameData->classIDs);
    extractBoundingBoxes(&frameData->rois);
    frameData->mask = extractImage(); // In this case, we can treat the assignment as atomic

#if 0
    // This is visualization code for debugging purposes
    static int a = 0;
    cv::putText(frameData->rgb, std::to_string(a), cv::Point(30,30), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0,255,255));
    cv::putText(frameData->mask, std::to_string(a), cv::Point(60,30), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(2));
    a++;

    const unsigned char colors[31][3] = {
        {0, 0, 0},     {0, 0, 255},     {255, 0, 0},   {0, 255, 0},     {255, 26, 184},  {255, 211, 0},   {0, 131, 246},  {0, 140, 70},
        {167, 96, 61}, {79, 0, 105},    {0, 255, 246}, {61, 123, 140},  {237, 167, 255}, {211, 255, 149}, {184, 79, 255}, {228, 26, 87},
        {131, 131, 0}, {0, 255, 149},   {96, 0, 43},   {246, 131, 17},  {202, 255, 0},   {43, 61, 0},     {0, 52, 193},   {255, 202, 131},
        {0, 43, 96},   {158, 114, 140}, {79, 184, 17}, {158, 193, 255}, {149, 158, 123}, {255, 123, 175}, {158, 8, 0}};
    auto getColor = [&colors](unsigned index) -> cv::Vec3b {
        return (index == 255) ? cv::Vec3b(255, 255, 255) : (cv::Vec3b)colors[index % 31];
    };
    cv::Mat vis(frameData->rgb.rows, frameData->rgb.cols, CV_8UC3);
    for (unsigned i = 0; i < frameData->rgb.total(); ++i) {
        vis.at<cv::Vec3b>(i) = getColor(frameData->mask.data[i]);
        vis.at<cv::Vec3b>(i) = 0.5 * vis.at<cv::Vec3b>(i) + 0.5 * frameData->rgb.at<cv::Vec3b>(i);
    }
    cv::imshow("out", vis);
    cv::waitKey(1);
#endif
}

void MaskRCNN::startThreadLoop(){
    if(thread.get_id() == std::thread::id())
        thread = std::thread(&MaskRCNN::loop, this);
}

void MaskRCNN::loop(){
    initialise();
    // Wait for first data
    while(pQueue->size()==0) {
        usleep(1e3);
        continue;
    }
    std::cout << "* MaskRCNN got first data -- starting loop." << std::endl;
    try{
        while(pQueue->size()){
            FrameDataPointer frame = pQueue->back(); // TODO technically not thread-safe, matters only at shutdown, however
            if(frame->mask.total() > 0) {
                // Probably the application is pausing
                //std::cout << "Pausing MASK-RCNN." << std::endl;
                continue;
            }
            TICK("MaskRCNN");
            executeSequential(frame);
            TOCK("MaskRCNN");
        }
        std::cout << "Shutting down MaskRCNN-Loop (no more frames)." << std::endl;
    } catch(...){
        threadException = std::current_exception();
        std::cout << "Shutting down MaskRCNN-Loop (ERROR)." << std::endl;
    }
}

PyObject *MaskRCNN::createArguments(cv::Mat rgbImage){
    assert(rgbImage.channels() == 3);
    npy_intp dims[3] = { rgbImage.rows, rgbImage.cols, 3 };
    return PyArray_SimpleNewFromData(3, dims, NPY_UINT8, rgbImage.data); // TODO Release?
}
