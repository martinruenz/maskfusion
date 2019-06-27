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
 */

#ifndef GUI_H_
#define GUI_H_

#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>
#include <map>
#include <GPUTexture.h>
#include <Utils/Intrinsics.h>
#include <Shaders/Shaders.h>
#include "Model/Buffers.h"
#include <vector>
#include <string>

#ifdef WITH_FREETYPE_GL_CPP
#define WITH_EIGEN
#include "freetype-gl-cpp/freetype-gl-cpp.h"
#endif

#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049

struct ModelInfo {
  // Constructor
  ModelInfo(int id, float conf) {
    this->id = id;
    std::string idString = std::to_string(id);
    confThreshold = new pangolin::Var<float>("oi.Model " + idString + " (conf-t)", conf, 0, 15);
  }

  // No copy constructor
  ModelInfo(const ModelInfo& m) = delete;

  // Move constructor (take ownership)
  ModelInfo(ModelInfo&& m) : id(m.id), confThreshold(m.confThreshold) {
    m.id = -1;
    m.confThreshold = nullptr;
  }
  virtual ~ModelInfo() {
    // This requires a patched Pangolin version and is now disabled by defaut.
    // Warning: Only call this destructor when terminating the application!
    // pangolin::RemoveVariable("oi.Model " + std::to_string(id) + " (conf-t)");
    delete confThreshold;
  }

  int id;
  pangolin::Var<float>* confThreshold;
};

class GUI {
 public:
  GUI(bool liveCap, bool showcaseMode) :
    window_width(1280 + widthPanel),
    window_height(980),
    showcaseMode(showcaseMode),
    s_cam(pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(window_width, window_height, 420, 420, window_width / 2.0f, window_height / 2.0f, 0.1, 1000),
                                      pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 1, pangolin::AxisNegY))),
    handler(s_cam)
  {
    pangolin::Params windowParams;

    windowParams.Set("SAMPLE_BUFFERS", 0);
    windowParams.Set("SAMPLES", 0);

    pangolin::CreateWindowAndBind("Main", window_width, window_height, windowParams);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    // Internally render at 3840x2160
    renderBuffer = new pangolin::GlRenderBuffer(1920, 1080);
    colorTexture = new GPUTexture(renderBuffer->width, renderBuffer->height, GL_RGBA32F, GL_RGBA, GL_FLOAT, true);

    colorFrameBuffer = new pangolin::GlFramebuffer;
    colorFrameBuffer->AttachColour(*colorTexture->texture);
    colorFrameBuffer->AttachDepth(*renderBuffer);

    colorProgram = std::shared_ptr<Shader>(
        loadProgramFromFile("draw_global_surface.vert", "draw_global_surface_phong.frag", "draw_global_surface.geom"));
    fxaaProgram = std::shared_ptr<Shader>(loadProgramFromFile("empty.vert", "fxaa.frag", "quad.geom"));

    pangolin::SetFullscreen(showcaseMode);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);

    pangolin::Display("cam").SetBounds(0, 1.0f, 0, 1.0f, -float(window_width) / window_height).SetHandler(&handler);
    pangolin::Display("ICP1").SetAspect(texture_aspect_ratio);
    pangolin::Display("ICP2").SetAspect(texture_aspect_ratio);
    pangolin::Display("ICP3").SetAspect(texture_aspect_ratio);
    pangolin::Display("ICP4").SetAspect(texture_aspect_ratio);

    //pangolin::Display("Model").SetAspect(width / height);

    std::vector<std::string> labels;
    labels.push_back(std::string("residual"));
    labels.push_back(std::string("threshold"));
    resLog.SetLabels(labels);

    resPlot = new pangolin::Plotter(&resLog, 0, 300, 0, 0.0005, 30, 0.5);
    resPlot->Track("$i");

    std::vector<std::string> labels2;
    labels2.push_back(std::string("inliers"));
    labels2.push_back(std::string("threshold"));
    inLog.SetLabels(labels2);

    inPlot = new pangolin::Plotter(&inLog, 0, 300, 0, 40000, 30, 0.5);
    inPlot->Track("$i");


    // Main-Side-Panel
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(widthPanel));

    // Objects-Panel
    pangolin::CreatePanel("oi").SetBounds(2 * heightTextures, 1.0, pangolin::Attach::Pix(widthPanel),
                                          pangolin::Attach::Pix(widthPanel * 2));

    // 1. row of textures
    pangolin::Display("multi-textures1")
            .SetBounds(heightTextures, 2 * heightTextures, pangolin::Attach::Pix(180), 1 - widthPlots)
            .SetLayout(pangolin::LayoutEqualHorizontal)
            .AddDisplay(pangolin::Display("P1"))
            .AddDisplay(pangolin::Display("P2"))
            .AddDisplay(pangolin::Display("P3"))
            .AddDisplay(pangolin::Display("P4"));

    // 2. row of textures
    pangolin::Display("multi-textures2")
            .SetBounds(0.0, heightTextures, pangolin::Attach::Pix(180), 1 - widthPlots)
            .SetLayout(pangolin::LayoutEqualHorizontal)
            .AddDisplay(pangolin::Display("ICP1"))
            .AddDisplay(pangolin::Display("ICP2"))
            .AddDisplay(pangolin::Display("ICP3"))
            .AddDisplay(pangolin::Display("ICP4"));

    // plots
    pangolin::Display("multi-plots")
            .SetBounds(0.0, 2 * heightTextures, 1 - widthPlots, 1.0)
            .SetLayout(pangolin::LayoutEqualHorizontal)
            .AddDisplay(*resPlot)  // plots
            .AddDisplay(*inPlot);

    if(showcaseMode){
        pangolin::Display("ui").Show(false);
        pangolin::Display("oi").Show(false);
        pangolin::Display("multi-textures1").Show(false);
        pangolin::Display("multi-textures2").Show(false);
        pangolin::Display("multi-plots").Show(false);
    }

    addMultiModelParameters();

    pause = new pangolin::Var<bool>("ui.Pause", false, true);
    step = new pangolin::Var<bool>("ui.Step", false, false);
    skip = new pangolin::Var<bool>("ui.Skip", false, false);
    saveCloud = new pangolin::Var<bool>("ui.Save cloud", false, false);
    savePoses = new pangolin::Var<bool>("ui.Save poses", false, false);
    // saveDepth = new pangolin::Var<bool>("ui.Save depth", false, false);
    saveView = new pangolin::Var<bool>("ui.Save view", false, false);
    reset = new pangolin::Var<bool>("ui.Reset", false, false);
    flipColors = new pangolin::Var<bool>("ui.Flip RGB", false, true);

    if (liveCap) {
      autoSettings = new pangolin::Var<bool>("ui.Auto Settings", true, true);
    } else {
      autoSettings = 0;
    }

    pyramid = new pangolin::Var<bool>("ui.Pyramid", true, true);
    so3 = new pangolin::Var<bool>("ui.SO(3)", true, true);
    frameToFrameRGB = new pangolin::Var<bool>("ui.Frame to frame RGB", false, true);
    fastOdom = new pangolin::Var<bool>("ui.Fast Odometry", false, true);
    rgbOnly = new pangolin::Var<bool>("ui.RGB only tracking", false, true);
    // confidenceThreshold = new pangolin::Var<float>("ui.Confidence threshold", 10.0, 0.0, 24.0);
    depthCutoff = new pangolin::Var<float>("ui.Depth cutoff", 4.0, 0.0, 20.0);
    icpWeight = new pangolin::Var<float>("ui.ICP weight", 20.0, 0.0, 100.0);
    outlierCoefficient = new pangolin::Var<float>("ui.Outlier Rejection", 0.1, 0, 5);

    followPose = new pangolin::Var<bool>("ui.Follow pose", true, true);
    drawRawCloud = new pangolin::Var<bool>("ui.Draw raw", false, true);
    drawFilteredCloud = new pangolin::Var<bool>("ui.Draw filtered", false, true);
    drawGlobalModel = new pangolin::Var<bool>("ui.Draw global model", true, true);
    drawObjectModels = new pangolin::Var<bool>("ui.Draw object models", true, true);
    drawUnstable = new pangolin::Var<bool>("ui.Draw unstable points", false, true);
    drawPoints = new pangolin::Var<bool>("ui.Draw points", false, true);
    drawColors = new pangolin::Var<bool>("ui.Draw colors", true, true);
    drawLabelColors = new pangolin::Var<bool>("ui.Draw label-color", false, true);
    drawPoseLog = new pangolin::Var<bool>("ui.Draw pose log", false, true);
    drawFxaa = new pangolin::Var<bool>("ui.Draw FXAA", false, true);
    drawWindow = new pangolin::Var<bool>("ui.Draw time window", false, true);
    drawNormals = new pangolin::Var<bool>("ui.Draw normals", false, true);
    drawTimes = new pangolin::Var<bool>("ui.Draw times", false, true);
    drawDefGraph = new pangolin::Var<bool>("ui.Draw deformation graph", false, true);
    drawFerns = new pangolin::Var<bool>("ui.Draw ferns", false, true);
    drawDeforms = new pangolin::Var<bool>("ui.Draw deformations", true, true);
    drawBoundingBoxes = new pangolin::Var<bool>("ui.Draw bounding-boxes", true, true);

    gpuMem = new pangolin::Var<int>("ui.GPU memory free", 0);

    totalPoints = new pangolin::Var<std::string>("ui.Total points", "0");
    totalNodes = new pangolin::Var<std::string>("ui.Total nodes", "0");
    totalFerns = new pangolin::Var<std::string>("ui.Total ferns", "0");
    totalDefs = new pangolin::Var<std::string>("ui.Total deforms", "0");
    totalFernDefs = new pangolin::Var<std::string>("ui.Total fern deforms", "0");

    trackInliers = new pangolin::Var<std::string>("ui.Inliers", "0");
    trackRes = new pangolin::Var<std::string>("ui.Residual", "0");
    logProgress = new pangolin::Var<std::string>("ui.Log", "0");

    if (showcaseMode) {
      pangolin::RegisterKeyPressCallback(' ', pangolin::SetVarFunctor<bool>("ui.Reset", true));
    }

    pangolin::RegisterKeyPressCallback('p', [&]() { pause->Ref().Set(!pause->Get()); });
    pangolin::RegisterKeyPressCallback('c', [&]() { drawColors->Ref().Set(!drawColors->Get()); });
    pangolin::RegisterKeyPressCallback('l', [&]() { drawLabelColors->Ref().Set(!drawLabelColors->Get()); });
    pangolin::RegisterKeyPressCallback('n', [&]() { drawNormals->Ref().Set(!drawNormals->Get()); });
    pangolin::RegisterKeyPressCallback('m', [&]() { enableMultiModel->Ref().Set(!enableMultiModel->Get()); });
    pangolin::RegisterKeyPressCallback('x', [&]() { drawFxaa->Ref().Set(!drawFxaa->Get()); });
    pangolin::RegisterKeyPressCallback('f', [&]() { followPose->Ref().Set(!followPose->Get()); });
    pangolin::RegisterKeyPressCallback('q', [&]() { savePoses->Ref().Set(true); });
    pangolin::RegisterKeyPressCallback('w', [&]() { saveCloud->Ref().Set(true); });
    pangolin::RegisterKeyPressCallback('e', [&]() { saveView->Ref().Set(true); });
    pangolin::RegisterKeyPressCallback('g', [&]() { drawGlobalModel->Ref().Set(!drawGlobalModel->Get()); });
    pangolin::RegisterKeyPressCallback('o', [&]() { drawObjectModels->Ref().Set(!drawObjectModels->Get()); });
    pangolin::RegisterKeyPressCallback('b', [&]() { drawBoundingBoxes->Ref().Set(!drawBoundingBoxes->Get()); });
#ifdef WITH_FREETYPE_GL_CPP
    textRenderer.init();
#endif
  }

  virtual ~GUI() {
    delete pause;
    delete reset;
    delete inPlot;
    delete resPlot;

    if (autoSettings) {
      delete autoSettings;
    }
    delete step;
    delete skip;
    delete saveCloud;
    delete savePoses;
    delete saveView;
    // delete saveDepth;
    delete trackInliers;
    delete trackRes;
    delete totalNodes;
    delete drawWindow;
    delete so3;
    delete totalFerns;
    delete totalDefs;
    delete depthCutoff;
    delete logProgress;
    delete drawTimes;
    delete drawFxaa;
    delete fastOdom;
    delete icpWeight;
    delete outlierCoefficient;
    delete pyramid;
    delete rgbOnly;
    delete totalFernDefs;
    delete drawFerns;
    delete followPose;
    delete drawDeforms;
    delete drawBoundingBoxes;
    delete drawRawCloud;
    delete totalPoints;
    delete frameToFrameRGB;
    delete flipColors;
    delete drawFilteredCloud;
    delete drawNormals;
    delete drawColors;
    delete drawLabelColors;
    delete drawPoseLog;
    delete drawGlobalModel;
    delete drawObjectModels;
    delete drawUnstable;
    delete drawPoints;
    delete drawDefGraph;
    delete gpuMem;

    delete renderBuffer;
    delete colorFrameBuffer;
    delete colorTexture;

    deleteCRFParameter();
    deleteBifoldParameters();
    deleteMultiModelParameters();
  }

  void addTextureColumn(std::vector<std::pair<std::string, std::shared_ptr<GPUTexture>>>& textures){
      if (showcaseMode) return;

      auto& view = pangolin::Display("multi-textures0")
              .SetBounds(2 * heightTextures,
                         1,
                         pangolin::Attach::Pix(widthPanel * 2),
                         pangolin::Attach::Pix(widthPanel * 2 + widthTextureColumn))
              .SetLayout(pangolin::LayoutEqualVertical);

      for (auto& name_texture : textures) {
          if(name_texture.second->draw) {
              // Add special case for debugging, if first letter == '!' show fullscreen
              if (name_texture.first.size() > 0 && name_texture.first[0] == '!') {
                  std::cout << "Notice, got debug texture (starting with '!')" << std::endl;
                  pangolin::Display(name_texture.first)
                          .SetAspect(texture_aspect_ratio)
                          .SetBounds(2 * heightTextures,
                                     1,
                                     pangolin::Attach::Pix(widthPanel * 2 + widthTextureColumn),
                                     pangolin::Attach::ReversePix(1));
              } else {
                  view.AddDisplay(pangolin::Display(name_texture.first).SetAspect(texture_aspect_ratio));
              }
          }
      }
  }


  void addMultiModelParameters(){
      enableMultiModel = new pangolin::Var<bool>("oi.Enable multiple models", true, true);
      enableSmartDelete = new pangolin::Var<bool>("oi.Delete deactivated", true, true);
      enableTrackAll = new pangolin::Var<bool>("oi.Track all models", false, true);
      minRelSizeNew = new pangolin::Var<float>("oi.Min-size new", 0.015, 0, 0.5);
      maxRelSizeNew = new pangolin::Var<float>("oi.Max-size new", 0.4, 0.3, 1);
      modelSpawnOffset = new pangolin::Var<unsigned>("oi.Model spawn offset", 22, 0, 100);
      modelDeactivateCnt = new pangolin::Var<unsigned>("oi.Deactivate model count", 10, 0, 100);
  }

  void deleteMultiModelParameters(){
      delete enableMultiModel;
      delete enableSmartDelete;
      delete enableTrackAll;
      delete modelSpawnOffset;
      delete modelDeactivateCnt;
      delete minRelSizeNew;
      delete maxRelSizeNew;
  }

  void addBifoldParameters(){
      bifoldBilateralSigmaDepth = new pangolin::Var<float>("oi.Filter sig-depth", 0.10, 0.001, 0.25);
      bifoldBilateralSigmaColor = new pangolin::Var<float>("oi.Filter sig-color", 40, 1, 80);
      bifoldBilateralSigmaLocation = new pangolin::Var<float>("oi.Filter sig-location", 5, 0.01, 10);
      bifoldBilateralRadius = new pangolin::Var<int>("oi.Filter Radius", 8, 1, 30);

      bifoldEdgeThreshold = new pangolin::Var<float>("oi.Threshold", 0.3, 0, 0.4);
      bifoldWeightDistance = new pangolin::Var<float>("oi.Weight Distance", 150, 0, 500);
      bifoldWeightConvexity = new pangolin::Var<float>("oi.Weight Convexity", 2.8, 0, 12);

      bifoldMorphEdgeIterations = new pangolin::Var<int>("oi.Morph Edge Iterations", 0, 0, 10);
      bifoldMorphEdgeRadius = new pangolin::Var<int>("oi.Morph Edge Radius", 1, 1, 5);
      bifoldMorphMaskIterations = new pangolin::Var<int>("oi.Morph Mask Iterations", 0, 0, 10);
      bifoldMorphMaskRadius = new pangolin::Var<int>("oi.Morph Mask Radius", 2, 1, 5);
      bifoldNonstaticThreshold = new pangolin::Var<float>("oi.Nonstatic Thresh", 0.5, 0, 1);
  }



  void addCRFParameter(){
      crfIterations = new pangolin::Var<unsigned>("oi.CRF iterations", 10, 0, 100);
      pairwiseRGBSTD = new pangolin::Var<float>("oi.CRF RGB std", 10, 0.05, 90);
      pairwiseDepthSTD = new pangolin::Var<float>("oi.CRF depth std", 0.9, 0.05, 5.0);
      pairwisePosSTD = new pangolin::Var<float>("oi.CRF pos std", 1.8, 0.05, 10.0);
      pairwiseAppearanceWeight = new pangolin::Var<float>("oi.CRF appearance weight", 7, 0.0, 50.0);
      pairwiseSmoothnessWeight = new pangolin::Var<float>("oi.CRF smoothness weight", 2, 0.0, 50.0);
      unaryErrorWeight = new pangolin::Var<float>("oi.Error weight", 75.0, 0.0, 400.0);
      unaryErrorK = new pangolin::Var<float>("oi.K-Error", 0.0375, 0.0001, 0.1);
      thresholdNew = new pangolin::Var<float>("oi.Thres New", 5.5, 0.0, 80.0);
  }

  void deleteBifoldParameters(){
      if(bifoldBilateralSigmaDepth) delete bifoldBilateralSigmaDepth;
      if(bifoldBilateralSigmaColor) delete bifoldBilateralSigmaColor;
      if(bifoldBilateralSigmaLocation) delete bifoldBilateralSigmaLocation;
      if(bifoldBilateralRadius) delete bifoldBilateralRadius;
      if(bifoldEdgeThreshold) delete bifoldEdgeThreshold;
      if(bifoldWeightConvexity) delete bifoldWeightConvexity;
      if(bifoldWeightDistance) delete bifoldWeightDistance;
      if(bifoldMorphEdgeIterations) delete bifoldMorphEdgeIterations;
      if(bifoldMorphEdgeRadius) delete bifoldMorphEdgeRadius;
      if(bifoldMorphMaskIterations) delete bifoldMorphMaskIterations;
      if(bifoldMorphMaskRadius) delete bifoldMorphMaskRadius;
      if(bifoldNonstaticThreshold) delete bifoldNonstaticThreshold;
  }

  void deleteCRFParameter(){
      if(pairwiseAppearanceWeight) delete pairwiseAppearanceWeight;
      if(pairwiseSmoothnessWeight) delete pairwiseSmoothnessWeight;
      if(pairwiseDepthSTD) delete pairwiseDepthSTD;
      if(pairwisePosSTD) delete pairwisePosSTD;
      if(pairwiseRGBSTD) delete pairwiseRGBSTD;
      if(thresholdNew) delete thresholdNew;
      if(unaryErrorK) delete unaryErrorK;
      if(unaryErrorWeight) delete unaryErrorWeight;
      if(crfIterations) delete crfIterations;
  }

  // Layout parameters
  const int widthPanel = 205;
  const float widthPlots = 0.35;
  const int widthTextureColumn = 350;
  const float heightTextures = 0.15;

  bool showcaseMode;
  int window_width;
  int window_height;

//  const float texture_width = 960;
//  const float texture_height = 540;
  const float texture_aspect_ratio = 960.0 / 540.0;

  void preCall() {
    //glClearColor(0.05 * !showcaseMode, 0.05 * !showcaseMode, 0.3 * !showcaseMode, 0.0f);
    glClearColor(1,1,1,1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    window_width = pangolin::DisplayBase().v.w;
    window_height = pangolin::DisplayBase().v.h;

    pangolin::Display("cam").Activate(s_cam);
  }

  inline void drawFrustum(const Eigen::Matrix4f& pose) {
    // if(showcaseMode) return;
    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
    K(0, 0) = Intrinsics::getInstance().fx();
    K(1, 1) = Intrinsics::getInstance().fy();
    K(0, 2) = Intrinsics::getInstance().cx();
    K(1, 2) = Intrinsics::getInstance().cy();

    Eigen::Matrix3f Kinv = K.inverse();
    pangolin::glDrawFrustum(Kinv, Resolution::getInstance().width(), Resolution::getInstance().height(), pose, 0.1f);
  }

  void displayImg(const std::string& id, GPUTexture* img) {
    if (showcaseMode) return;

    glDisable(GL_DEPTH_TEST);

    pangolin::Display(id).Activate();
    img->texture->RenderToViewport(true);

    glEnable(GL_DEPTH_TEST);
  }

  void addModel(int id, float confThres) { modelInfos.emplace_back(id, confThres); }

  void displayEmpty(const std::string& id) {
    if (showcaseMode) return;

    glDisable(GL_DEPTH_TEST);

    pangolin::Display(id).Activate();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // GLfloat sq_vert[] = { -1,-1,  1,-1,  1, 1,  -1, 1 };
    // glVertexPointer(2, GL_FLOAT, 0, sq_vert);
    // glEnableClientState(GL_VERTEX_ARRAY);

    // GLfloat sq_tex[]  = { 0,0,  1,0,  1,1,  0,1  };
    // glTexCoordPointer(2, GL_FLOAT, 0, sq_tex);
    // glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    // glEnable(GL_TEXTURE_2D);
    // Bind();

    glLineWidth(1);
    glBegin(GL_LINES);
    glColor3f(1, 0, 0);
    glVertex3f(-1, -1, -1);
    glVertex3f(1, 1, 1);
    glEnd();

    // img->texture->RenderToViewport(true);

    glEnable(GL_DEPTH_TEST);
  }

  void saveColorImage(const std::string& path) { pangolin::SaveWindowOnRender(path); }

  void postCall() {
    GLint cur_avail_mem_kb = 0;
    glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &cur_avail_mem_kb);

    int memFree = cur_avail_mem_kb / 1024;

    gpuMem->operator=(memFree);

    pangolin::FinishFrame();

    glFinish();
  }

  void drawFXAA(const Eigen::Matrix4f& matViewProjection, const Eigen::Matrix4f& pose, pangolin::OpenGlMatrix mv,
                std::list<std::shared_ptr<Model>>& models, const int time, const int timeDelta, const bool invertNormals) {
    // First pass computes positions, colors and normals per pixel
    colorFrameBuffer->Bind();

    glPushAttrib(GL_VIEWPORT_BIT);

    glViewport(0, 0, renderBuffer->width, renderBuffer->height);

    glClearColor(0.05 * !showcaseMode, 0.05 * !showcaseMode, 0.3 * !showcaseMode, 0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    colorProgram->Bind();

    for (auto& modelPointer : models) {
      Eigen::Matrix4f mvp = matViewProjection;
      if (modelPointer->getID() != 0) mvp = mvp * pose * modelPointer->getPose().inverse();

      colorProgram->setUniform(Uniform("MVP", mvp));
      colorProgram->setUniform(Uniform("threshold", modelPointer->getConfidenceThreshold()));
      colorProgram->setUniform(Uniform("time", time));
      colorProgram->setUniform(Uniform("timeDelta", timeDelta));
      colorProgram->setUniform(Uniform("signMult", invertNormals ? 1.0f : -1.0f));
      colorProgram->setUniform(Uniform("colorType", (drawNormals->Get() ? 1 : drawColors->Get() ? 2 : drawTimes->Get() ? 3 : 0)));
      colorProgram->setUniform(Uniform("unstable", drawUnstable->Get()));
      colorProgram->setUniform(Uniform("drawWindow", drawWindow->Get()));

      Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
      // This is for the point shader
      colorProgram->setUniform(Uniform("pose", pose));

      Eigen::Matrix4f modelView = mv;

      Eigen::Vector3f lightpos = modelView.topRightCorner(3, 1);

      colorProgram->setUniform(Uniform("lightpos", lightpos));

      glBindBuffer(GL_ARRAY_BUFFER, modelPointer->getModelBuffer().dataBuffer);

      glEnableVertexAttribArray(0);
      glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

      glEnableVertexAttribArray(1);
      glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 1));

      glEnableVertexAttribArray(2);
      glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

      glDrawTransformFeedback(GL_POINTS, modelPointer->getModelBuffer().stateObject);

      glDisableVertexAttribArray(0);
      glDisableVertexAttribArray(1);
      glDisableVertexAttribArray(2);
      glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    colorFrameBuffer->Unbind();

    colorProgram->Unbind();

    glPopAttrib();

    fxaaProgram->Bind();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, colorTexture->texture->tid);

    Eigen::Vector2f resolution(renderBuffer->width, renderBuffer->height);

    fxaaProgram->setUniform(Uniform("tex", 0));
    fxaaProgram->setUniform(Uniform("resolution", resolution));

    glDrawArrays(GL_POINTS, 0, 1);

    fxaaProgram->Unbind();

    glBindFramebuffer(GL_READ_FRAMEBUFFER, colorFrameBuffer->fbid);

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    glBlitFramebuffer(0, 0, renderBuffer->width, renderBuffer->height, 0, 0, window_width, window_height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, 0);

    glFinish();
  }

  void drawFXAAOLD(pangolin::OpenGlMatrix mvp, pangolin::OpenGlMatrix mv, const OutputBuffer& model, const float threshold, const int time,
                   const int timeDelta, const bool invertNormals) {
    // First pass computes positions, colors and normals per pixel
    colorFrameBuffer->Bind();

    glPushAttrib(GL_VIEWPORT_BIT);

    glViewport(0, 0, renderBuffer->width, renderBuffer->height);

    glClearColor(0.05 * !showcaseMode, 0.05 * !showcaseMode, 0.3 * !showcaseMode, 0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    colorProgram->Bind();

    colorProgram->setUniform(Uniform("MVP", mvp));
    colorProgram->setUniform(Uniform("threshold", threshold));
    colorProgram->setUniform(Uniform("time", time));
    colorProgram->setUniform(Uniform("timeDelta", timeDelta));
    colorProgram->setUniform(Uniform("signMult", invertNormals ? 1.0f : -1.0f));
    colorProgram->setUniform(Uniform("colorType", (drawNormals->Get() ? 1 : drawColors->Get() ? 2 : drawTimes->Get() ? 3 : 0)));
    colorProgram->setUniform(Uniform("unstable", drawUnstable->Get()));
    colorProgram->setUniform(Uniform("drawWindow", drawWindow->Get()));

    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    // This is for the point shader
    colorProgram->setUniform(Uniform("pose", pose));

    Eigen::Matrix4f modelView = mv;

    Eigen::Vector3f lightpos = modelView.topRightCorner(3, 1);

    colorProgram->setUniform(Uniform("lightpos", lightpos));

    glBindBuffer(GL_ARRAY_BUFFER, model.dataBuffer);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 1));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

    glDrawTransformFeedback(GL_POINTS, model.stateObject);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    colorFrameBuffer->Unbind();

    colorProgram->Unbind();

    glPopAttrib();

    fxaaProgram->Bind();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, colorTexture->texture->tid);

    Eigen::Vector2f resolution(renderBuffer->width, renderBuffer->height);

    fxaaProgram->setUniform(Uniform("tex", 0));
    fxaaProgram->setUniform(Uniform("resolution", resolution));

    glDrawArrays(GL_POINTS, 0, 1);

    fxaaProgram->Unbind();

    glBindFramebuffer(GL_READ_FRAMEBUFFER, colorFrameBuffer->fbid);

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    glBlitFramebuffer(0, 0, renderBuffer->width, renderBuffer->height, 0, 0, window_width, window_height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, 0);

    glFinish();
  }
#ifdef WITH_FREETYPE_GL_CPP
  void addLabelTexts(const std::vector<std::string>& labels, const std::vector<ftgl::vec4>& colors){
      assert(labels.size() == colors.size() || colors.size() == 1);
      for (unsigned i = 0; i < labels.size(); ++i) {
          ftgl::Markup markup = textRenderer.createMarkup("DejaVu Sans", 32, colors[colors.size() == 1 ? 0 : i]);
          ftgl::FreetypeGlText label = textRenderer.createText(labels[i], markup);
          label.setScalingFactor(0.0011);
          textLabels.push_back({std::move(label), std::move(markup)});
      }
  }
#endif

  pangolin::Handler3D handler;
  pangolin::Var<bool> *pause, *step, *skip, *savePoses, *saveView, *saveCloud,
      //* saveDepth,
      *reset, *flipColors, *rgbOnly, *enableMultiModel, *enableSmartDelete, *enableTrackAll, *pyramid, *so3, *frameToFrameRGB, *fastOdom, *followPose,
      *drawRawCloud, *drawFilteredCloud, *drawNormals, *autoSettings, *drawDefGraph, *drawColors, *drawPoseLog, *drawLabelColors, *drawFxaa,
      *drawGlobalModel, *drawObjectModels, *drawUnstable, *drawPoints, *drawTimes, *drawFerns, *drawDeforms, *drawWindow, *drawBoundingBoxes;
  pangolin::Var<int>* gpuMem;
  pangolin::Var<std::string> *totalPoints, *totalNodes, *totalFerns, *totalDefs, *totalFernDefs, *trackInliers, *trackRes, *logProgress;

  pangolin::Var<float> *depthCutoff, *icpWeight, *outlierCoefficient;

  // Model related
  pangolin::Var<int>* numModels;
  std::vector<ModelInfo> modelInfos;
  pangolin::Var<unsigned> *modelSpawnOffset, *modelDeactivateCnt;

  // Segmentation
  pangolin::Var<float> *minRelSizeNew, *maxRelSizeNew;

  // CRF
  pangolin::Var<float> *pairwiseRGBSTD = nullptr;
  pangolin::Var<float> *pairwiseDepthSTD = nullptr;
  pangolin::Var<float> *pairwisePosSTD = nullptr;
  pangolin::Var<float> *pairwiseAppearanceWeight = nullptr;
  pangolin::Var<float> *pairwiseSmoothnessWeight = nullptr;
  pangolin::Var<float> *thresholdNew = nullptr;
  pangolin::Var<float> *unaryErrorWeight = nullptr;
  pangolin::Var<float> *unaryErrorK = nullptr;
  pangolin::Var<unsigned> *crfIterations = nullptr;

  // Bifold
  pangolin::Var<float> *bifoldBilateralSigmaDepth = nullptr;
  pangolin::Var<float> *bifoldBilateralSigmaColor = nullptr;
  pangolin::Var<float> *bifoldBilateralSigmaLocation = nullptr;
  pangolin::Var<int> *bifoldBilateralRadius = nullptr;
  pangolin::Var<float> *bifoldEdgeThreshold = nullptr;
  pangolin::Var<float> *bifoldWeightDistance = nullptr;
  pangolin::Var<float> *bifoldWeightConvexity = nullptr;
  pangolin::Var<float> *bifoldNonstaticThreshold = nullptr;
  pangolin::Var<int> *bifoldMorphEdgeIterations = nullptr;
  pangolin::Var<int> *bifoldMorphEdgeRadius = nullptr;
  pangolin::Var<int> *bifoldMorphMaskIterations = nullptr;
  pangolin::Var<int> *bifoldMorphMaskRadius = nullptr;

  pangolin::DataLog resLog, inLog;
  pangolin::Plotter *resPlot, *inPlot;

  pangolin::OpenGlRenderState s_cam;

  pangolin::GlRenderBuffer* renderBuffer;
  pangolin::GlFramebuffer* colorFrameBuffer;
  GPUTexture* colorTexture;
  std::shared_ptr<Shader> colorProgram;
  std::shared_ptr<Shader> fxaaProgram;

#ifdef WITH_FREETYPE_GL_CPP
  ftgl::FreetypeGl textRenderer;
  std::vector<std::pair<ftgl::FreetypeGlText, ftgl::Markup>> textLabels;
#endif
};

#endif /* GUI_H_ */
