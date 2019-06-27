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

#pragma once

#include "Utils/Macros.h"
#include "Utils/RGBDOdometry.h"
#include "Utils/Resolution.h"
#include "Utils/Intrinsics.h"
#include "Utils/Stopwatch.h"
#include "Callbacks.h"
#include "Shaders/Shaders.h"
#include "Shaders/ComputePack.h"
#include "Shaders/FeedbackBuffer.h"
#include "Shaders/FillIn.h"
#include "Model/Deformation.h"
#include "Model/Model.h"
#include "Model/ModelProjection.h"
#include "Model/GlobalProjection.h"
#include "Ferns.h"
#include "PoseMatch.h"
#include "FrameData.h"
#include "Segmentation/Segmentation.h"

#include <list>
#include <iomanip>
#include <memory>
#include <pangolin/gl/glcuda.h>

class MaskFusion {
 public:
  MaskFusion(int timeDelta = 200, int countThresh = 35000, float errThresh = 5e-05, float covThresh = 1e-05,
           bool closeLoops = true, bool iclnuim = false, bool reloc = false, float photoThresh = 115,
           float initConfidenceGlobal = 4, float initConfidenceObject = 2, float depthCut = 3, float icpThresh = 10,
           bool fastOdom = false, float fernThresh = 0.3095, bool so3 = true, bool frameToFrameRGB = false,
           unsigned modelSpawnOffset = 20, Model::MatchingType matchingType = Model::MatchingType::Drost,
           Segmentation::Method segmentationMethod = Segmentation::Method::MASK_FUSION,
           const std::string& exportDirectory = "", bool exportSegmentationResults = false, bool usePrecomputedMasksOnly = false, unsigned frameQueueSize = 0);

  virtual ~MaskFusion();

  void preallocateModels(unsigned count);

  SegmentationResult performSegmentation(FrameDataPointer frame);

  /**
       * Process an rgb/depth map pair
       * @param frame Frame data (rgb, depth, time)
       * @param inPose optional input SE3 pose (if provided, we don't attempt to perform tracking)
       * @param weightMultiplier optional full frame fusion weight
       * @param bootstrap if true, use inPose as a pose guess rather than replacement
       * @return returns true if a pause might be interesting, can be ignored without hesitation
       */
  bool processFrame(FrameDataPointer frame, const Eigen::Matrix4f* inPose = 0, const float weightMultiplier = 1.f,
                    const bool bootstrap = false);

  /**
       * Predicts the current view of the scene, updates the [vertex/normal/image]Tex() members
       * of the indexMap class
       */
  void predict();

  /**
       * This class contains all of the predicted renders
       * @return reference
       */
  ModelProjection& getIndexMap();

  /**
       * This class contains the surfel map
       * @return
       */
  std::shared_ptr<Model> getBackgroundModel();

  std::list<std::shared_ptr<Model>>& getModels();

  /**
       * This class contains the fern keyframe database
       * @return
       */
  Ferns& getFerns();

  /**
       * This class contains the local deformation graph
       * @return
       */
  Deformation& getLocalDeformation();

  /**
       * This is the map of raw input textures (you can display these)
       * @return
       */
  std::vector<std::pair<std::string, std::shared_ptr<GPUTexture>>>& getDrawableTextures();

  /**
       * This is the list of deformation constraints
       * @return
       */
  const std::vector<PoseMatch>& getPoseMatches();

  /**
       * This is the tracking class, if you want access
       * @return
       */
  const RGBDOdometry& getModelToModel();

  /**
       * The point fusion confidence threshold
       * @return
       */
  const float& getConfidenceThreshold();

  /**
       * If you set this to true we just do 2.5D RGB-only Lucas–Kanade tracking (no fusion)
       * @param val
       */
  void setRgbOnly(const bool& val);

  /**
       * Weight for ICP in tracking
       * @param val if 100, only use depth for tracking, if 0, only use RGB. Best value is 10
       */
  void setIcpWeight(const float& val);

  void setOutlierCoefficient(const float& val);

  /**
       * Whether or not to use a pyramid for tracking
       * @param val default is true
       */
  void setPyramid(const bool& val);

  /**
       * Controls the number of tracking iterations
       * @param val default is false
       */
  void setFastOdom(const bool& val);

  /**
       * Turns on or off SO(3) alignment bootstrapping
       * @param val
       */
  void setSo3(const bool& val);

  /**
       * Turns on or off frame to frame tracking for RGB
       * @param val
       */
  void setFrameToFrameRGB(const bool& val);

  /**
       * Raw data fusion confidence threshold
       * @param val default value is 10, but you can play around with this
       */
  void setConfidenceThreshold(const float& val);

  /**
       * Threshold for sampling new keyframes
       * @param val default is some magic value, change at your own risk
       */
  void setFernThresh(const float& val);

  /**
       * Cut raw depth input off at this point
       * @param val default is 3 meters
       */
  void setDepthCutoff(const float& val);

  /**
       * Returns whether or not the camera is lost, if relocalisation mode is on
       * @return
       */
  const bool& getLost();

  /**
       * Get the internal clock value of the fusion process
       * @return monotonically increasing integer value (not real-world time)
       */
  const int& getTick();

  /**
       * Get the time window length for model matching
       * @return
       */
  const int& getTimeDelta();

  /**
       * Cheat the clock, only useful for multisession/log fast forwarding
       * @param val control time itself!
       */
  void setTick(const int& val);

  /**
       * Internal maximum depth processed, this is defaulted to 20 (for rescaling depth buffers)
       * @return
       */
  const float& getMaxDepthProcessed();

  /**
       * The current global camera pose estimate
       * @return SE3 pose
       */
  const Eigen::Matrix4f& getCurrPose();

  /**
       * The number of local deformations that have occurred
       * @return
       */
  const int& getDeforms();

  /**
       * The number of global deformations that have occured
       * @return
       */
  const int& getFernDeforms();

  // void setExportSegmentationDirectory(const std::string& path) { exportSegDir = path; }

  void setMfBilatSigmaDepth(float val);
  void setMfBilatSigmaColor(float val);
  void setMfBilatSigmaLocation(float val);
  void setMfBilatRadius(int val);
  void setMfMorphEdgeRadius(int val);
  void setMfMorphEdgeIterations(int val);
  void setMfMorphMaskRadius(int val);
  void setMfMorphMaskIterations(int val);
  void setMfThreshold(float val);
  void setMfWeightDistance(float val);
  void setMfWeightConvexity(float val);
  void setMfNonstaticThreshold(float val);
  void setTrackableClassIds(const std::set<int>& ids);

  void setModelSpawnOffset(const unsigned& val);
  void setModelDeactivateCount(const unsigned& val);
  void setCfPairwiseSigmaRGB(const float& val);
  void setCfPairwiseSigmaPosition(const float& val);
  void setCfPairwiseSigmaDepth(const float& val);
  void setCfPairwiseWeightAppearance(const float& val);
  void setCfPairwiseWeightSmoothness(const float& val);
  void setCfThresholdNew(const float& val);
  void setCfUnaryWeightError(const float& val);
  void setCfIteration(const unsigned& val);
  void setCfUnaryKError(const float& val);
  void setNewModelMinRelativeSize(const float& val);
  void setNewModelMaxRelativeSize(const float& val);
  void setEnableMultipleModels(bool val) { enableMultipleModels = val; }
  void setTrackAllModels(bool val) { trackAllModels = val; }
  void setEnableSmartModelDelete(bool val) { enableSmartModelDelete = val; }
  // void setCfUnaryWeightErrorBackground(const float& val);
  // void setCfUnaryWeightConfBackground(const float& val);

  /**
       * These are the vertex buffers computed from the raw input data
       * Each of these buffers stores one vertex per pixel
       * @return can be rendered
       */
  std::map<std::string, FeedbackBuffer*>& getFeedbackBuffers();

  /**
       * Calculate the above for the current frame (only done on the first frame normally)
       */
  void computeFeedbackBuffers();

  /**
   * Saves out a .ply mesh file of the current model
   */
  void savePly();

  void exportPoses();

  /**
       * Renders a normalised view of the input metric depth for displaying as an OpenGL texture
       * (this is stored under textures[GPUTexture::DEPTH_NORM]
       * @param minVal minimum depth value to render
       * @param maxVal maximum depth value to render
       */
  void normaliseDepth(const float& minVal, const float& maxVal);

  /**
       * Renders a colorised view of the segmentation (masks)
       * (this is stored under textures[GPUTexture::MASK_COLOR]
       */
  void coloriseMasks();

  // Listeners

  /// Called when a new model is created
  inline void addNewModelListener(const ModelListener& listener) { newModelListeners.addListener(listener); }

  /// Called when a model becomes inactive
  inline void addInactiveModelListener(const ModelListener& listener) { inactiveModelListeners.addListener(listener); }

  // Here be dragons
 private:
  void spawnObjectModel();
  bool redetectModels(const FrameData& frame, const SegmentationResult& segmentationResult);
  void moveNewModelToList();
  ModelListIterator inactivateModel(const ModelListIterator& it);

  unsigned char getNextModelID(bool assign = false);

  void createTextures();
  void createCompute();
  void createFeedbackBuffers();

  void filterDepth();

  // Should raw data (of last frame) be added to the rendered vertexTexture / normalTexture / imageTexture (from last pose)?
  bool requiresFillIn(ModelPointer model, float ratio = 0.75f);

  void processFerns();

 private:
  ModelList models;  // also contains static environment (first model)
  ModelList inactiveModels;
  ModelList preallocatedModels;        // Since some systems are slow in allocating new models, allow to preallocate models for later use.
  ModelPointer newModel;               // Stored and added to list as soon as frame is processed
  std::shared_ptr<Model> globalModel;  // static environment
  unsigned char nextID = 0;
  Segmentation labelGenerator;

  GlobalProjection globalProjection;

  Model::MatchingType modelMatchingType;

  CallbackBuffer<std::shared_ptr<Model>> newModelListeners;
  CallbackBuffer<std::shared_ptr<Model>> inactiveModelListeners;

  RGBDOdometry modelToModel;

  // TODO move to model?
  Ferns ferns;
  Deformation localDeformation;
  Deformation globalDeformation;

  std::shared_ptr<GPUTexture> textureRGB;
  std::shared_ptr<GPUTexture> textureDepthMetric;
  std::shared_ptr<GPUTexture> textureDepthMetricFiltered;
  std::shared_ptr<GPUTexture> textureDepthNorm;
  std::shared_ptr<GPUTexture> textureMask;
  std::shared_ptr<GPUTexture> textureMaskColor;
  std::vector<std::pair<std::string, std::shared_ptr<GPUTexture>>> drawableTextures;

  std::map<std::string, ComputePack*> computePacks;
  std::map<std::string, FeedbackBuffer*> feedbackBuffers;

  std::queue<FrameDataPointer> frameQueue;
  unsigned queueLength;

  int tick;
  const int timeDelta;
  const int icpCountThresh;
  const float icpErrThresh;
  const float covThresh;

  int deforms;
  int fernDeforms;
  const int consSample;
  GPUResize resize;

  std::vector<PoseMatch> poseMatches;
  std::vector<Deformation::Constraint> relativeCons;

  // std::vector<std::pair<int64_t, Eigen::Matrix4f> > poseGraph;
  // std::vector<int64_t> poseLogTimes;

  // Scaled down textures
  Img<Eigen::Matrix<unsigned char, 3, 1>> imageBuff;  // imageTex (only used for fillIn)
  Img<Eigen::Vector4f> consBuff;                      // vertexTex
  Img<unsigned short> timesBuff;                      // oldTimeTex

  const bool closeLoops;
  const bool iclnuim;

  const bool reloc;
  bool lost;
  bool lastFrameRecovery;
  int trackingCount;
  const float maxDepthProcessed;

  bool enableMultipleModels = true;
  bool trackAllModels = true;
  bool enableSmartModelDelete = true;
  bool enableRedetection = false;
  bool enableModelMerging = false;
  bool enableSpawnSubtraction = true;
  bool enablePoseLogging = true;
  bool rgbOnly;
  float icpWeight;
  bool pyramid;
  bool fastOdom;
  float initConfThresGlobal;
  float initConfThresObject;
  float fernThresh;
  bool so3;
  bool frameToFrameRGB;
  float depthCutoff;
  unsigned modelDeactivateCount = 10;   // deactivate model, when not seen for this many frames FIXME unused
  unsigned modelKeepMinSurfels = 4000;  // Only keep deactivated models with at least this many surfels
  float modelKeepConfThreshold = 0.3;
  unsigned modelSpawnOffset;  // setting
  unsigned spawnOffset = 0;   // current value

  std::set<int> trackableClassIds;

  bool exportSegmentation;
  std::string exportDir;

  CameraModel cudaIntrinsics;
};
