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


#include "GlobalProjection.h"
#include "Model.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Utils/OpenGLErrorHandling.h"

GlobalProjection::GlobalProjection(int w, int h)
    : program(loadProgramFromFile("splat_models.vert", "combo_splat_models.frag")),
      renderbuffer(w, h),
      texDepth(w, h, GL_R16F, GL_RED, GL_FLOAT),
      texID(w, h, GL_R8UI, GL_RED_INTEGER, GL_UNSIGNED_BYTE) {

    framebuffer.AttachColour(*texDepth.texture);
    framebuffer.AttachColour(*texID.texture);
    framebuffer.AttachDepth(renderbuffer);

    idBuffer = cv::Mat(h, w, CV_8UC1);
    depthBuffer = cv::Mat(h, w, CV_32FC1);
}

GlobalProjection::~GlobalProjection() {}

void GlobalProjection::project(const std::list<std::shared_ptr<Model>>& models, int time, int maxTime, int timeDelta, float depthCutoff){

    if(!models.size()) return;

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);

    framebuffer.Bind();
    glPushAttrib(GL_VIEWPORT_BIT);
    glViewport(0, 0, renderbuffer.width, renderbuffer.height);

    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    program->Bind();
    program->setUniform(Uniform("cols", (float)Resolution::getInstance().cols()));
    program->setUniform(Uniform("rows", (float)Resolution::getInstance().rows()));
    program->setUniform(Uniform("maxDepth", depthCutoff));
    program->setUniform(Uniform("confThreshold", (float)12));
    program->setUniform(Uniform("time", time));
    program->setUniform(Uniform("maxTime", maxTime));
    program->setUniform(Uniform("timeDelta", timeDelta));

    for(auto& model : models){

        const Eigen::Matrix4f& pose = model->getPose();
        const OutputBuffer& modelBuffer = model->getModelBuffer();

        Eigen::Matrix4f t_inv = pose.inverse();
        Eigen::Vector4f cam(Intrinsics::getInstance().cx(), Intrinsics::getInstance().cy(), Intrinsics::getInstance().fx(),
                            Intrinsics::getInstance().fy());

        program->setUniform(Uniform("t_inv", t_inv));
        program->setUniform(Uniform("cam", cam));
        program->setUniform(Uniform("modelID", (int)model->getID()));

        glBindBuffer(GL_ARRAY_BUFFER, modelBuffer.dataBuffer);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 1));

        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

        glDrawTransformFeedback(GL_POINTS, modelBuffer.stateObject);  // RUN GPU-PASS

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glDisableVertexAttribArray(2);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    framebuffer.Unbind();
    program->Unbind();

    glDisable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_POINT_SPRITE);

    glPopAttrib();
    glFinish();

    checkGLErrors();
}

void GlobalProjection::downloadDirect(){
    texID.downloadTexture(idBuffer.data);

    // TODO FIX
    //texDepth.downloadTexture(depthBuffer.data);

//    cv::Mat testID(480, 640, CV_8UC3);
//    cv::Mat testDepth(480, 640, CV_8UC1);
//    const unsigned char colors[31][3] = {
//        {0, 0, 0},     {0, 0, 255},     {255, 0, 0},   {0, 255, 0},     {255, 26, 184},  {255, 211, 0},   {0, 131, 246},  {0, 140, 70},
//        {167, 96, 61}, {79, 0, 105},    {0, 255, 246}, {61, 123, 140},  {237, 167, 255}, {211, 255, 149}, {184, 79, 255}, {228, 26, 87},
//        {131, 131, 0}, {0, 255, 149},   {96, 0, 43},   {246, 131, 17},  {202, 255, 0},   {43, 61, 0},     {0, 52, 193},   {255, 202, 131},
//        {0, 43, 96},   {158, 114, 140}, {79, 184, 17}, {158, 193, 255}, {149, 158, 123}, {255, 123, 175}, {158, 8, 0}};
//    auto getColor = [&colors](unsigned index) -> cv::Vec3b {
//        return (index == 255) ? cv::Vec3b(255, 255, 255) : (cv::Vec3b)colors[index % 31];
//    };
//    for(unsigned i=0; i < idBuffer.total(); i++){
//        testID.at<cv::Vec3b>(i) = getColor(idBuffer.at<unsigned char>(i));
//        testDepth.at<uchar>(i) = 30 * depthBuffer.at<float>(i);
//    }
//    cv::imshow("TEST_ID", testID);
//    cv::imshow("DEPTH_ID", testDepth);
//    cv::waitKey(1);
}
