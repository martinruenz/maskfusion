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

#include "MaskFusion.h"
#include <thread>

//#define LOG_TICKS

MaskFusion::MaskFusion(int timeDelta, int countThresh, float errThresh, float covThresh, bool closeLoops,
                   bool iclnuim, bool reloc, float photoThresh, float initConfidenceGlobal,
                   float initConfidenceObject, float depthCut, float icpThresh, bool fastOdom,
                   float fernThresh, bool so3, bool frameToFrameRGB, unsigned modelSpawnOffset,
                   Model::MatchingType matchingType, Segmentation::Method segmentationMethod,
                   const std::string& exportDirectory, bool exportSegmentationResults, bool usePrecomputedMasksOnly, unsigned frameQueueSize)
    : globalProjection(Resolution::getInstance().width(), Resolution::getInstance().height()),
      modelMatchingType(matchingType),
      newModelListeners(0),
      inactiveModelListeners(0),
      modelToModel(Resolution::getInstance().width(), Resolution::getInstance().height(), Intrinsics::getInstance().cx(),
                   Intrinsics::getInstance().cy(), Intrinsics::getInstance().fx(), Intrinsics::getInstance().fy()),
      ferns(500.0f, depthCut * 1000.0f, photoThresh),
      queueLength((usePrecomputedMasksOnly || segmentationMethod != Segmentation::Method::MASK_FUSION) ? 0 : frameQueueSize),
      tick(1),
      timeDelta(timeDelta),
      icpCountThresh(countThresh),
      icpErrThresh(errThresh),
      covThresh(covThresh),
      deforms(0),
      fernDeforms(0),
      consSample(20),
      resize(Resolution::getInstance().width(), Resolution::getInstance().height(), Resolution::getInstance().width() / consSample,
             Resolution::getInstance().height() / consSample),
      imageBuff(Resolution::getInstance().rows() / consSample, Resolution::getInstance().cols() / consSample),
      consBuff(Resolution::getInstance().rows() / consSample, Resolution::getInstance().cols() / consSample),
      timesBuff(Resolution::getInstance().rows() / consSample, Resolution::getInstance().cols() / consSample),
      closeLoops(closeLoops),
      iclnuim(iclnuim),
      reloc(reloc),
      lost(false),
      lastFrameRecovery(false),
      trackingCount(0),
      maxDepthProcessed(20.0f),
      rgbOnly(false),
      icpWeight(icpThresh),
      pyramid(true),
      fastOdom(fastOdom),
      initConfThresGlobal(initConfidenceGlobal),
      initConfThresObject(initConfidenceObject),
      fernThresh(fernThresh),
      so3(so3),
      frameToFrameRGB(frameToFrameRGB),
      depthCutoff(depthCut),
      modelSpawnOffset(modelSpawnOffset),
      exportSegmentation(exportSegmentationResults),
      exportDir(exportDirectory),
      cudaIntrinsics(Intrinsics::getInstance().fx(), Intrinsics::getInstance().fy(), Intrinsics::getInstance().cx(), Intrinsics::getInstance().cy())
{
    createTextures();
    createCompute();
    createFeedbackBuffers();

    labelGenerator.init(Resolution::getInstance().width(), Resolution::getInstance().height(), segmentationMethod, cudaIntrinsics, textureRGB, textureDepthMetric, usePrecomputedMasksOnly, &globalProjection, frameQueueSize ? &frameQueue : nullptr);
    auto segTextures = labelGenerator.getDrawableTextures();
    drawableTextures.insert(drawableTextures.end(), segTextures.begin(), segTextures.end());
    globalModel = std::make_shared<Model>(getNextModelID(true), initConfidenceGlobal, true, true, enablePoseLogging);
    models.push_back(globalModel);

    // This will create an issue in the destructor, use shared pointers or similar.
    //textures["GLOBAL_MODEL"] = globalModel->getIndexMap().getSplatImageTex();

    Stopwatch::getInstance().setCustomSignature(12431231);

    // Select GPU and print GPU info, as detected by OpenCV
    int gpuSlam = MASKFUSION_GPU_SLAM;
    if (gpuSlam >= 0) cudaSetDevice(gpuSlam);
#if 0
    if (cv::ocl::haveOpenCL()){
        cv::ocl::Context context;
        if (context.create(cv::ocl::Device::TYPE_GPU)){
            int numDevices = context.ndevices();
            std::cout << "OpenCV detected " << numDevices << " GPU(s):" << std::endl;
            for (int i = 0; i < numDevices; i++){
                cv::ocl::Device device = context.device(i);
                std::cout << i << ": " << device.name()
                          << "\n (available: " << (device.available() ? "yes" : "no")
                          << ", image-support: " << (device.imageSupport() ? "yes" : "no")
                          << ", version: " << device.OpenCL_C_Version() << ")" << std::endl;
            }
            if (setSlamGPU){
                if(gpuSlam < numDevices) cv::ocl::Device(context.device(gpuSlam));
                else throw std::runtime_error("Invalid '-gpuSLAM' parameter, maybe your OpenCV version only supports 1 GPU.\n"
                                              "Consider setting 'WITH_OPENCL=OFF' in OpenCV-cmake.");
                // Note: Your OpenCV version might enforce numDevices==1.
                // In this case, consider setting 'WITH_OPENCL=OFF' in OpenCV-cmake.
            }
        }
    }
#endif

    std::cout << "Initialised multi-object fusion (main-thread: " << std::this_thread::get_id() << ")"
                 "\n- The background model can have up to " << Model::MAX_VERTICES_GLOBAL << " surfel (" << Model::TEXTURE_DIMENSION_GLOBAL << "x" << Model::TEXTURE_DIMENSION_GLOBAL << ")"
                 "\n- Object models can have up to " << Model::MAX_VERTICES_OBJECT << " surfel (" << Model::TEXTURE_DIMENSION_OBJECT << "x" << Model::TEXTURE_DIMENSION_OBJECT << ")"
                 "\n- Using GPU " << ((gpuSlam >= 0) ? std::to_string(gpuSlam) : "unspecified") << " for SLAM system and GPU " << MASKFUSION_GPUS_MASKRCNN << " for MaskRCNN"
                 "\n- Using frame-queue of size: " << queueLength << std::endl;
}

MaskFusion::~MaskFusion() {
    if (iclnuim) {
        savePly();
    }

    for (std::map<std::string, ComputePack*>::iterator it = computePacks.begin(); it != computePacks.end(); ++it) {
        delete it->second;
    }

    computePacks.clear();

    for (std::map<std::string, FeedbackBuffer*>::iterator it = feedbackBuffers.begin(); it != feedbackBuffers.end(); ++it) {
        delete it->second;
    }

    feedbackBuffers.clear();

    cudaCheckError();

    labelGenerator.cleanup();
}

void MaskFusion::preallocateModels(unsigned count) {
    unsigned id0 = getNextModelID();
    for (unsigned i = 0; i < count; ++i)
        preallocatedModels.push_back(
                    std::make_shared<Model>(id0++, initConfThresObject, false, true, enablePoseLogging, modelMatchingType));
}

SegmentationResult MaskFusion::performSegmentation(FrameDataPointer frame) {
    return labelGenerator.performSegmentation(models, frame, getNextModelID(), spawnOffset >= modelSpawnOffset);
}

void MaskFusion::createTextures() {
    textureRGB = std::make_shared<GPUTexture>(Resolution::getInstance().width(), Resolution::getInstance().height(), GL_RGBA, GL_RGB, GL_UNSIGNED_BYTE, true, true);
    textureDepthMetric = std::make_shared<GPUTexture>(Resolution::getInstance().width(), Resolution::getInstance().height(), GL_R32F, GL_RED, GL_FLOAT, false, true);
    textureDepthMetricFiltered = std::make_shared<GPUTexture>(Resolution::getInstance().width(), Resolution::getInstance().height(), GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT, false, true);
    textureMask = std::make_shared<GPUTexture>(Resolution::getInstance().width(), Resolution::getInstance().height(),
                                               GL_R8UI,         // GL_R8, GL_R8UI, GL_R8I internal
                                               GL_RED_INTEGER,  // GL_RED, GL_RED_INTEGER // format
                                               GL_UNSIGNED_BYTE, false, true);

    // Visualisation only textures
    textureDepthNorm = std::make_shared<GPUTexture>(Resolution::getInstance().width(), Resolution::getInstance().height(), GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT, true);
    textureMaskColor = std::make_shared<GPUTexture>(Resolution::getInstance().width(), Resolution::getInstance().height(), GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, true);

    drawableTextures.push_back({"RGB", textureRGB});
    drawableTextures.push_back({"DepthNorm", textureDepthNorm});
    drawableTextures.push_back({"MaskColor", textureMaskColor});
}

void MaskFusion::createCompute() {
    computePacks[ComputePack::FILTER] = new ComputePack(loadProgramFromFile("empty.vert", "depth_bilateral_metric.frag", "quad.geom"),
                                                        textureDepthMetricFiltered->texture);
    // Visualisation only
    computePacks[ComputePack::NORM_DEPTH] =
            new ComputePack(loadProgramFromFile("empty.vert", "depth_norm.frag", "quad.geom"), textureDepthNorm->texture);

    computePacks[ComputePack::COLORISE_MASKS] =
            new ComputePack(loadProgramFromFile("empty.vert", "int_to_color.frag", "quad.geom"), textureMaskColor->texture);
}

void MaskFusion::createFeedbackBuffers() {
    feedbackBuffers[FeedbackBuffer::RAW] =
            new FeedbackBuffer(loadProgramGeomFromFile("vertex_feedback.vert", "vertex_feedback.geom"));  // Used to render raw depth data
    feedbackBuffers[FeedbackBuffer::FILTERED] = new FeedbackBuffer(loadProgramGeomFromFile("vertex_feedback.vert", "vertex_feedback.geom"));
}

void MaskFusion::computeFeedbackBuffers() {
    TICK("feedbackBuffers");
    feedbackBuffers[FeedbackBuffer::RAW]->compute(textureRGB->texture, textureDepthMetric->texture, tick,
                                                  maxDepthProcessed);

    feedbackBuffers[FeedbackBuffer::FILTERED]->compute(textureRGB->texture,
                                                       textureDepthMetricFiltered->texture, tick, maxDepthProcessed);
    TOCK("feedbackBuffers");
}

bool MaskFusion::processFrame(FrameDataPointer frame, const Eigen::Matrix4f* inPose, const float weightMultiplier, const bool bootstrap) {
    assert(frame->depth.type() == CV_32FC1);
    assert(frame->rgb.type() == CV_8UC3);
    assert(frame->timestamp >= 0);
    TICK("Run");

    frameQueue.push(frame);
    if(frameQueue.size() < queueLength) return 0;
    frame = frameQueue.front();
    frameQueue.pop();

    // Upload RGB to graphics card
    textureRGB->texture->Upload(frame->rgb.data, GL_RGB, GL_UNSIGNED_BYTE);

    TICK("Preprocess");

    textureDepthMetric->texture->Upload((float*)frame->depth.data, GL_LUMINANCE, GL_FLOAT);
    filterDepth();

    // if(frame.mask) {
    //    // Use ground-truth segmentation if provided (TODO: Overwritten at the moment)
    //    textureMask->texture->Upload(frame.mask, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_BYTE);
    //} else
    if (!enableMultipleModels) {
        // If the support for multiple objects is deactivated, segment everything as background (static scene).
        const long size = Resolution::getInstance().width() * Resolution::getInstance().height();
        unsigned char* data = new unsigned char[size];
        memset(data, 0, size);
        textureMask->texture->Upload(data, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_BYTE);
        delete[] data;
    }

    TOCK("Preprocess");

    // First run
    if (tick == 1) {
        computeFeedbackBuffers();
        globalModel->initialise(*feedbackBuffers[FeedbackBuffer::RAW], *feedbackBuffers[FeedbackBuffer::FILTERED]);
        globalModel->getFrameOdometry().initFirstRGB(textureRGB.get());
    } else {
        bool trackingOk = true;

        // Regular execution, false if pose is provided by user
        if (bootstrap || !inPose) {
            Model::generateCUDATextures(textureDepthMetricFiltered.get(), textureMask.get(), cudaIntrinsics, depthCutoff);

            TICK("odom");
            ModelListIterator itr = models.begin();
            (*itr)->performTracking(frameToFrameRGB,
                                    rgbOnly, icpWeight,
                                    pyramid,
                                    fastOdom,
                                    so3,
                                    maxDepthProcessed,
                                    textureRGB.get(),
                                    frame->timestamp,
                                    requiresFillIn(*itr));


            for(++itr;itr!=models.end();itr++){
                ModelPointer& mp = (*itr);
                bool trackable = trackableClassIds.empty() || trackableClassIds.count(mp->getClassID());

                if(((*itr)->isNonstatic() || trackAllModels) && trackable){
                    Eigen::Matrix4f t = (*itr)->performTracking(frameToFrameRGB, rgbOnly, icpWeight, pyramid, fastOdom, so3, maxDepthProcessed, textureRGB.get(),
                                                                frame->timestamp, requiresFillIn(*itr));

                    // Don't allow big jumps (remove jumping models)
                    float d = t.topRightCorner(3, 1).norm(); // Hack, do something reasonable
                    if(d > 0.2){
                        std::cout << "Removing model due to movement." << std::endl;
                        itr = inactivateModel(itr);
                    }
                } else {
                    mp->updateStaticPose(globalModel->getPose()); // cam->cam_0=object_0 (cam_0->object_0 = identity)
                }
            }

            TOCK("odom");

            if (bootstrap) {
                assert(inPose);
                globalModel->overridePose(globalModel->getPose() * (*inPose));
            }

            trackingOk = !reloc || globalModel->getFrameOdometry().lastICPError < 1e-04;

            if (enableMultipleModels) {

                globalProjection.project(models, tick, tick, timeDelta, depthCutoff);
                globalProjection.downloadDirect();

                auto getMaxDepth = [](const SegmentationResult::ModelData& data) -> float { return data.depthMean + data.depthStd * 1.2; };

                if (spawnOffset < modelSpawnOffset) spawnOffset++;

                SegmentationResult segmentationResult = performSegmentation(frame);
                textureMask->texture->Upload(segmentationResult.fullSegmentation.data, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_BYTE);

                if (exportSegmentation) {
                    cv::Mat output;
                    cv::threshold(segmentationResult.fullSegmentation, output, 254, 255, cv::THRESH_TOZERO_INV);
                    cv::imwrite(exportDir + "Segmentation" + std::to_string(tick) + ".png", output);
                }

#ifdef EXPORT_OBJECT_MASKS
                    // Export object masks
                    cv::Mat projectedIDs = globalProjection.getProjectedModelIDs();
                    cv::Mat mask = projectedIDs > 0;
                    cv::imwrite(exportDir + "ObjectMasks" + std::to_string(tick) + ".png", mask);
                    cv::imshow( "Object masks", mask);
#endif

                // Spawn new model
                if (segmentationResult.hasNewLabel) {
                    const SegmentationResult::ModelData& newModelData = segmentationResult.modelData.back();
                    std::cout << "New label detected ("
                              << newModelData.boundingBox.left << ","
                              << newModelData.boundingBox.top << " "
                              << newModelData.boundingBox.right << ","
                              << newModelData.boundingBox.bottom << ") - try relocating..." << std::endl;

                    // New model
                    std::cout << "Found new model." << std::endl;

                    spawnObjectModel();
                    spawnOffset = 0;

                    newModel->setMaxDepth(getMaxDepth(newModelData));
                    newModel->setClassID(newModelData.classID);

                    moveNewModelToList();
                }

                // Set max-depth for all models
                ModelList::iterator it = models.begin();
                for (unsigned i = 1; i < models.size(); i++) {
                    ModelPointer& m = *++it;
                    m->setMaxDepth(getMaxDepth(segmentationResult.modelData[i]));
                }

                // Initialise new model data
                if (segmentationResult.hasNewLabel) {
                    ModelPointer& nm = models.back();
                    nm->predictIndices(tick, maxDepthProcessed, timeDelta);

                    nm->fuse(tick, textureRGB.get(), textureMask.get(), textureDepthMetric.get(),
                             textureDepthMetricFiltered.get(), maxDepthProcessed, 100);

                    // newModel->predictIndices(tick, maxDepthProcessed, timeDelta);

                    std::vector<float> test;
                    nm->clean(tick, test, timeDelta, maxDepthProcessed, false, textureDepthMetricFiltered.get(),
                              textureMask.get());

                    enableSpawnSubtraction = false;
                    if (enableSpawnSubtraction) {
                        globalModel->eraseErrorGeometry(textureDepthMetricFiltered.get());
                    }
                }

                // Check for nonstatic objects
                it = models.begin();
                for (unsigned i = 1; i < models.size(); i++) {
                    ModelPointer& m = *++it;
                    SegmentationResult::ModelData& md = segmentationResult.modelData[i];
                }

                // increase confidence of object-models
                it = models.begin();
                for (unsigned i = 1; i < models.size(); i++) {
                    ModelPointer& m = *++it;
                    float factor = std::min(4.5f, m->getAge() / 25.0f);
                    m->setConfidenceThreshold(factor);
                }
            }

            if (reloc) {
                if (!lost) {
                    Eigen::MatrixXd covariance = globalModel->getFrameOdometry().getCovariance();

                    for (int i = 0; i < 6; i++) {
                        if (covariance(i, i) > 1e-04) {
                            trackingOk = false;
                            break;
                        }
                    }

                    if (!trackingOk) {
                        trackingCount++;

                        if (trackingCount > 10) {
                            lost = true;
                        }
                    } else {
                        trackingCount = 0;
                    }
                } else if (lastFrameRecovery) {
                    Eigen::MatrixXd covariance = globalModel->getFrameOdometry().getCovariance();

                    for (int i = 0; i < 6; i++) {
                        if (covariance(i, i) > 1e-04) {
                            trackingOk = false;
                            break;
                        }
                    }

                    if (trackingOk) {
                        lost = false;
                        trackingCount = 0;
                    }

                    lastFrameRecovery = false;
                }
            }  // reloc

        }  // regular
        else {
            globalModel->overridePose(*inPose);
        }

        std::vector<Ferns::SurfaceConstraint> constraints;

        predict();

        Eigen::Matrix4f recoveryPose = globalModel->getPose();

        if (closeLoops) {
            lastFrameRecovery = false;

            TICK("Ferns::findFrame");
            recoveryPose = ferns.findFrame(constraints, globalModel->getPose(), globalModel->getFillInVertexTexture(),
                                           globalModel->getFillInNormalTexture(), globalModel->getFillInImageTexture(), tick, lost);
            TOCK("Ferns::findFrame");
        }

        std::vector<float> rawGraph;

        bool fernAccepted = false;

        if (closeLoops && ferns.lastClosest != -1) {
            if (lost) {
                globalModel->overridePose(recoveryPose);
                lastFrameRecovery = true;
            } else {
                for (size_t i = 0; i < constraints.size(); i++)
                    globalDeformation.addConstraint(constraints.at(i).sourcePoint, constraints.at(i).targetPoint, tick,
                                                    ferns.frames.at(ferns.lastClosest)->srcTime, true);

                for (size_t i = 0; i < relativeCons.size(); i++) globalDeformation.addConstraint(relativeCons.at(i));

                assert(0);  // FIXME, input pose-graph again.
                if (globalDeformation.constrain(ferns.frames, rawGraph, tick, true, /*poseGraph,*/ true)) {
                    globalModel->overridePose(recoveryPose);
                    poseMatches.push_back(PoseMatch(ferns.lastClosest, ferns.frames.size(), ferns.frames.at(ferns.lastClosest)->pose,
                                                    globalModel->getPose(), constraints, true));
                    fernDeforms += rawGraph.size() > 0;
                    fernAccepted = true;
                }
            }
        }

        // If we didn't match to a fern
        if (!lost && closeLoops && rawGraph.size() == 0) {
            // Only predict old view, since we just predicted the current view for the ferns (which failed!)
            TICK("IndexMap::INACTIVE");
            globalModel->combinedPredict(maxDepthProcessed, 0, tick - timeDelta, timeDelta, ModelProjection::INACTIVE);
            TOCK("IndexMap::INACTIVE");

            // WARNING initICP* must be called before initRGB*
            // RGBDOdometry& modelToModel = globalModel->getModelToModelOdometry();
            ModelProjection& indexMap = globalModel->getIndexMap();
            assert(0); // This function is not implemented anymore FIXME
            //modelToModel.initICPModel(indexMap.getOldVertexTex(), indexMap.getOldNormalTex(), maxDepthProcessed, globalModel->getPose());
            //modelToModel.initRGBModel(indexMap.getOldImageTex());
            //modelToModel.initICP(indexMap.getSplatVertexConfTex(), indexMap.getSplatNormalTex(), maxDepthProcessed);
            //modelToModel.initRGB(indexMap.getSplatImageTex());

            Eigen::Vector3f trans = globalModel->getPose().topRightCorner(3, 1);
            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = globalModel->getPose().topLeftCorner(3, 3);

            modelToModel.getIncrementalTransformation(trans, rot, false, 10, pyramid, fastOdom, false, 0, 0);

            Eigen::MatrixXd covar = modelToModel.getCovariance();
            bool covOk = true;

            for (int i = 0; i < 6; i++) {
                if (covar(i, i) > covThresh) {
                    covOk = false;
                    break;
                }
            }

            Eigen::Matrix4f estPose = Eigen::Matrix4f::Identity();

            estPose.topRightCorner(3, 1) = trans;
            estPose.topLeftCorner(3, 3) = rot;

            if (covOk && modelToModel.lastICPCount > icpCountThresh && modelToModel.lastICPError < icpErrThresh) {
                resize.vertex(indexMap.getSplatVertexConfTex(), consBuff);
                resize.time(indexMap.getOldTimeTex(), timesBuff);

                for (int i = 0; i < consBuff.cols; i++) {
                    for (int j = 0; j < consBuff.rows; j++) {
                        if (consBuff.at<Eigen::Vector4f>(j, i)(2) > 0 && consBuff.at<Eigen::Vector4f>(j, i)(2) < maxDepthProcessed &&
                                timesBuff.at<unsigned short>(j, i) > 0) {
                            Eigen::Vector4f worldRawPoint =
                                    globalModel->getPose() * Eigen::Vector4f(consBuff.at<Eigen::Vector4f>(j, i)(0), consBuff.at<Eigen::Vector4f>(j, i)(1),
                                                                             consBuff.at<Eigen::Vector4f>(j, i)(2), 1.0f);

                            Eigen::Vector4f worldModelPoint =
                                    globalModel->getPose() * Eigen::Vector4f(consBuff.at<Eigen::Vector4f>(j, i)(0), consBuff.at<Eigen::Vector4f>(j, i)(1),
                                                                             consBuff.at<Eigen::Vector4f>(j, i)(2), 1.0f);

                            constraints.push_back(Ferns::SurfaceConstraint(worldRawPoint, worldModelPoint));

                            localDeformation.addConstraint(worldRawPoint, worldModelPoint, tick, timesBuff.at<unsigned short>(j, i), deforms == 0);
                        }
                    }
                }

                std::vector<Deformation::Constraint> newRelativeCons;

                assert(0);
                if (localDeformation.constrain(ferns.frames, rawGraph, tick, false, /*poseGraph,*/ false, &newRelativeCons)) {
                    poseMatches.push_back(
                                PoseMatch(ferns.frames.size() - 1, ferns.frames.size(), estPose, globalModel->getPose(), constraints, false));

                    deforms += rawGraph.size() > 0;

                    globalModel->overridePose(estPose);

                    for (size_t i = 0; i < newRelativeCons.size(); i += newRelativeCons.size() / 3) {
                        relativeCons.push_back(newRelativeCons.at(i));
                    }
                }
            }
        }

        if (!rgbOnly && trackingOk && !lost) {
            TICK("indexMap");
            for (auto model : models) model->predictIndices(tick, maxDepthProcessed, timeDelta);
            TOCK("indexMap");

            for (auto model : models) {
                model->fuse(tick, textureRGB.get(), textureMask.get(), textureDepthMetric.get(),
                            textureDepthMetricFiltered.get(), depthCutoff, weightMultiplier);
            }

            TICK("indexMap");
            for (auto model : models) model->predictIndices(tick, maxDepthProcessed, timeDelta);
            TOCK("indexMap");

            // If we're deforming we need to predict the depth again to figure out which
            // points to update the timestamp's of, since a deformation means a second pose update
            // this loop
            if (rawGraph.size() > 0 && !fernAccepted) {
                globalModel->getIndexMap().synthesizeDepth(globalModel->getPose(), globalModel->getModelBuffer(), maxDepthProcessed, initConfThresGlobal,
                                                           tick, tick - timeDelta, std::numeric_limits<unsigned short>::max());
            }

            for (auto model : models) {
                model->clean(tick, rawGraph, timeDelta, maxDepthProcessed, fernAccepted, textureDepthMetricFiltered.get(),
                             textureMask.get());
            }
        }
    }

    // Update index-map textures
    predict();

    if (!lost) {
        // processFerns(); FIXME
        tick++;
    }

    moveNewModelToList();

    bool first = true;

    for (auto model : models) {
        if (model->isLoggingPoses()) {
            auto pose = first ? globalModel->getPose() :                          // cam->world
                                globalModel->getPose() * model->getPose().inverse();  // obj->world

            Eigen::Vector3f transObject = pose.topRightCorner(3, 1);
            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rotObject = pose.topLeftCorner(3, 3);
            Eigen::Quaternionf q(rotObject);

            Eigen::Matrix<float, 7, 1> p;
            p << transObject(0), transObject(1), transObject(2), q.x(), q.y(), q.z(), q.w();
#ifdef LOG_TICKS
            model->getPoseLog().push_back({tick-1, p}); //Log ticks
#else
            model->getPoseLog().push_back({frame->timestamp, p});  // Log timestamps
#endif


        }
        first = false;
        model->incrementAge();
        // std::cout << "Model " << model->getID() << " has " << model->lastCount() << " surfels." << std::endl;
    }

    TOCK("Run");

    return false;
}

void MaskFusion::processFerns() {
    TICK("Ferns::addFrame");
    ferns.addFrame(globalModel->getFillInImageTexture(), globalModel->getFillInVertexTexture(), globalModel->getFillInNormalTexture(),
                   globalModel->getPose(), tick, fernThresh);
    TOCK("Ferns::addFrame");
}

void MaskFusion::predict() {
    TICK("IndexMap::ACTIVE");

    for (auto& model : models) {
        // Predict textures based on the current pose estimate
        model->combinedPredict(maxDepthProcessed, lastFrameRecovery ? 0 : tick, tick, timeDelta, ModelProjection::ACTIVE);

        // Generate textures that fill holes in predicted data with raw data (if enabled by model, currently only global model)
        model->performFillIn(textureRGB.get(), textureDepthMetricFiltered.get(), frameToFrameRGB, lost);
    }

    TOCK("IndexMap::ACTIVE");
}

bool MaskFusion::requiresFillIn(ModelPointer model, float ratio) {
    if (!model->allowsFillIn()) return false;

    TICK("autoFill");
    resize.image(model->getRGBProjection(), imageBuff);
    int sum = 0;

    // TODO do this faster
    for (int i = 0; i < imageBuff.rows; i++) {
        for (int j = 0; j < imageBuff.cols; j++) {
            sum += imageBuff.at<Eigen::Matrix<unsigned char, 3, 1>>(i, j)(0) > 0 &&
                                                                   imageBuff.at<Eigen::Matrix<unsigned char, 3, 1>>(i, j)(1) > 0 && imageBuff.at<Eigen::Matrix<unsigned char, 3, 1>>(i, j)(2) > 0;
        }
    }
    TOCK("autoFill");

    // Checks whether less than ratio (75%) of the pixels are set
    return float(sum) / float(imageBuff.rows * imageBuff.cols) < ratio;
}

void MaskFusion::filterDepth() {
    std::vector<Uniform> uniforms;
    uniforms.push_back(Uniform("cols", (float)Resolution::getInstance().cols()));
    uniforms.push_back(Uniform("rows", (float)Resolution::getInstance().rows()));
    uniforms.push_back(Uniform("maxD", depthCutoff));
    computePacks[ComputePack::FILTER]->compute(textureDepthMetric->texture,
                                               &uniforms);  // Writes to TEXTURE_DEPTH_METRIC_FILTERED
}

void MaskFusion::normaliseDepth(const float& minVal, const float& maxVal) {
    std::vector<Uniform> uniforms;
    uniforms.push_back(Uniform("maxVal", maxVal));
    uniforms.push_back(Uniform("minVal", minVal));
    computePacks[ComputePack::NORM_DEPTH]->compute(textureDepthMetric->texture,
                                                   &uniforms);  // Writes to TEXTURE_DEPTH_NORM
}

void MaskFusion::coloriseMasks() {
    computePacks[ComputePack::COLORISE_MASKS]->compute(textureMask->texture);  // Writes to TEXTURE_MASK_COLOR
}

void MaskFusion::spawnObjectModel() {
    assert(!newModel);
    if (preallocatedModels.size()) {
        newModel = preallocatedModels.front();
        preallocatedModels.pop_front();
        getNextModelID(true);
    } else {
        newModel = std::make_shared<Model>(getNextModelID(true), initConfThresObject, false, true, enablePoseLogging, modelMatchingType);
    }
    newModel->getFrameOdometry().initFirstRGB(textureRGB.get());
    newModel->makeStatic(globalModel->getPose());
    //newModel->enableFiltering();
    //newModel->makeNonStatic();
}

bool MaskFusion::redetectModels(const FrameData& frame, const SegmentationResult& segmentationResult) {
    // [Removed code]
    return false;
}

void MaskFusion::moveNewModelToList() {
    if (newModel) {
        models.push_back(newModel);
        newModelListeners.callListenersDirect(newModel);
        newModel.reset();
    }
}

ModelListIterator MaskFusion::inactivateModel(const ModelListIterator& it) {
    std::shared_ptr<Model> m = *it;
    std::cout << "Deactivating model... ";
    if (!enableSmartModelDelete || (m->lastCount() >= modelKeepMinSurfels && m->getConfidenceThreshold() > modelKeepConfThreshold)) {
        std::cout << "keeping data";
        // [Removed code]
        inactiveModels.push_back(m);
    } else {
        std::cout << "deleting data";
    }
    std::cout << ". Surfels: " << m->lastCount() << " confidence threshold: " << m->getConfidenceThreshold() << std::endl;

    inactiveModelListeners.callListenersDirect(m);
    return --models.erase(it);
}

unsigned char MaskFusion::getNextModelID(bool assign) {
    unsigned char next = nextID;
    if (assign) {
        if (models.size() == 256)
            throw std::range_error(
                    "getNextModelID(): Maximum amount of models is "
                    "already in use (256).");
        while (true) {
            nextID++;
            bool isOccupied = false;
            for (auto& m : models)
                if (nextID == m->getID()) isOccupied = true;
            if (!isOccupied) break;
        }
    }
    return next;
}

void MaskFusion::savePly() {
    std::cout << "Exporting PLYs..." << std::endl;

    auto exportModelPLY = [this](ModelPointer& model) {

        std::string filename = exportDir + "cloud-" + std::to_string(model->getID()) + ".ply";
        std::cout << "Storing PLY-cloud to " << filename << std::endl;

        // Open file
        std::ofstream fs;
        fs.open(filename.c_str());

        Model::SurfelMap surfelMap = model->downloadMap();
        surfelMap.countValid(model->getConfidenceThreshold());

        std::cout << "Extarcted " << surfelMap.numValid << " out of " << surfelMap.numPoints << " points." << std::endl;

        // Write header
        fs << "ply";
        fs << "\nformat "
           << "binary_little_endian"
           << " 1.0";

        // Vertices
        fs << "\nelement vertex " << surfelMap.numValid;
        fs << "\nproperty float x"
              "\nproperty float y"
              "\nproperty float z";

        fs << "\nproperty uchar red"
              "\nproperty uchar green"
              "\nproperty uchar blue";

        fs << "\nproperty float nx"
              "\nproperty float ny"
              "\nproperty float nz";

        fs << "\nproperty float radius";

        fs << "\nend_header\n";

        // Close the file
        fs.close();

        // Open file in binary appendable
        std::ofstream fpout(filename.c_str(), std::ios::app | std::ios::binary);

        Eigen::Vector4f center(0, 0, 0, 0);

#ifdef EXPORT_GLOBAL_PLY
        Eigen::Matrix4f gP = globalModel->getPose();
        Eigen::Matrix4f Tp = gP * model->getPose().inverse();
        Eigen::Matrix4f Tn = Tn.inverse().transpose();
#endif

        for (unsigned int i = 0; i < surfelMap.numPoints; i++) {
            Eigen::Vector4f pos = (*surfelMap.data)[(i * 3) + 0];
            float conf = pos[3];
            pos[3] = 1;

            if (conf > model->getConfidenceThreshold()) {
                Eigen::Vector4f col = (*surfelMap.data)[(i * 3) + 1];
                Eigen::Vector4f nor = (*surfelMap.data)[(i * 3) + 2];
                center += pos;
                float radius = nor[3];
                nor[3] = 0;
#ifdef EXPORT_GLOBAL_PLY
                pos = Tp * pos;
                nor = Tn * nor;
#endif

                nor[0] *= -1;
                nor[1] *= -1;
                nor[2] *= -1;

                float value;
                memcpy(&value, &pos[0], sizeof(float));
                fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

                memcpy(&value, &pos[1], sizeof(float));
                fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

                memcpy(&value, &pos[2], sizeof(float));
                fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

                unsigned char r = int(col[0]) >> 16 & 0xFF;
                unsigned char g = int(col[0]) >> 8 & 0xFF;
                unsigned char b = int(col[0]) & 0xFF;

                fpout.write(reinterpret_cast<const char*>(&r), sizeof(unsigned char));
                fpout.write(reinterpret_cast<const char*>(&g), sizeof(unsigned char));
                fpout.write(reinterpret_cast<const char*>(&b), sizeof(unsigned char));

                memcpy(&value, &nor[0], sizeof(float));
                fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

                memcpy(&value, &nor[1], sizeof(float));
                fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

                memcpy(&value, &nor[2], sizeof(float));
                fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));

                memcpy(&value, &radius, sizeof(float));
                fpout.write(reinterpret_cast<const char*>(&value), sizeof(float));
            }
        }

        center /= surfelMap.numValid;
        std::cout << "Exported model with center: \n" << center << std::endl;

        // Close file
        fs.close();
    };

    for (auto& m : models) exportModelPLY(m);
}

void MaskFusion::exportPoses() {
    std::cout << "Exporting poses..." << std::endl;

    auto exportModelPoses = [&](ModelList list) {
        for (auto& m : list) {
            if (!m->isLoggingPoses()) continue;
            std::string filename = exportDir + "poses-" + std::to_string(m->getID()) + ".txt";
            std::cout << "Storing poses to " << filename << std::endl;

            std::ofstream fs;
            fs.open(filename.c_str());
            fs << std::fixed << std::setprecision(6);
            auto poseLog = m->getPoseLog();
            for (auto& p : poseLog) {
#ifdef LOG_TICKS
                fs << p.ts;
#else
                fs << double(p.ts) * 1e-6;
#endif
                for (int i = 0; i < p.p.size(); ++i) fs << " " << p.p(i);
                fs << "\n";
            }

            fs.close();
        }
    };

    exportModelPoses(models);
    exportModelPoses(inactiveModels);
}

// Sad times ahead
ModelProjection& MaskFusion::getIndexMap() { return globalModel->getIndexMap(); }

ModelPointer MaskFusion::getBackgroundModel() { return globalModel; }

ModelList& MaskFusion::getModels() { return models; }

Ferns& MaskFusion::getFerns() { return ferns; }

Deformation& MaskFusion::getLocalDeformation() { return localDeformation; }

std::vector<std::pair<std::string, std::shared_ptr<GPUTexture>>>& MaskFusion::getDrawableTextures(){ return drawableTextures; }

const std::vector<PoseMatch>& MaskFusion::getPoseMatches() { return poseMatches; }

const RGBDOdometry& MaskFusion::getModelToModel() { return modelToModel; }

void MaskFusion::setRgbOnly(const bool& val) { rgbOnly = val; }

void MaskFusion::setIcpWeight(const float& val) { icpWeight = val; }

void MaskFusion::setOutlierCoefficient(const float& val) { Model::GPUSetup::getInstance().outlierCoefficient = val; }

void MaskFusion::setPyramid(const bool& val) { pyramid = val; }

void MaskFusion::setFastOdom(const bool& val) { fastOdom = val; }

void MaskFusion::setSo3(const bool& val) { so3 = val; }

void MaskFusion::setFrameToFrameRGB(const bool& val) { frameToFrameRGB = val; }

void MaskFusion::setModelSpawnOffset(const unsigned& val) { modelSpawnOffset = val; }

void MaskFusion::setModelDeactivateCount(const unsigned& val) { modelDeactivateCount = val; }

void MaskFusion::setCfPairwiseSigmaRGB(const float& val) { labelGenerator.getCfSegmentationPerformer()->setPairwiseSigmaRGB(val); }
void MaskFusion::setCfPairwiseSigmaPosition(const float& val) { labelGenerator.getCfSegmentationPerformer()->setPairwiseSigmaPosition(val); }
void MaskFusion::setCfPairwiseSigmaDepth(const float& val) { labelGenerator.getCfSegmentationPerformer()->setPairwiseSigmaDepth(val); }
void MaskFusion::setCfPairwiseWeightAppearance(const float& val) { labelGenerator.getCfSegmentationPerformer()->setPairwiseWeightAppearance(val); }
void MaskFusion::setCfPairwiseWeightSmoothness(const float& val) { labelGenerator.getCfSegmentationPerformer()->setPairwiseWeightSmoothness(val); }
void MaskFusion::setCfThresholdNew(const float& val) { labelGenerator.getCfSegmentationPerformer()->setUnaryThresholdNew(val); }
void MaskFusion::setCfUnaryWeightError(const float& val) { labelGenerator.getCfSegmentationPerformer()->setUnaryWeightError(val); }
void MaskFusion::setCfIteration(const unsigned& val) { labelGenerator.getCfSegmentationPerformer()->setIterationsCRF(val); }
void MaskFusion::setCfUnaryKError(const float& val) { labelGenerator.getCfSegmentationPerformer()->setUnaryKError(val); }
void MaskFusion::setNewModelMinRelativeSize(const float& val) { labelGenerator.getSegmentationPerformer()->setNewModelMinRelativeSize(val); }
void MaskFusion::setNewModelMaxRelativeSize(const float& val) { labelGenerator.getSegmentationPerformer()->setNewModelMaxRelativeSize(val); }

void MaskFusion::setMfNonstaticThreshold(float val){ labelGenerator.getMfSegmentationPerformer()->nonstaticThreshold = val; }
void MaskFusion::setMfBilatSigmaDepth(float val) { labelGenerator.getMfSegmentationPerformer()->bilatSigmaDepth = val; }
void MaskFusion::setMfBilatSigmaColor(float val) { labelGenerator.getMfSegmentationPerformer()->bilatSigmaColor = val; }
void MaskFusion::setMfBilatSigmaLocation(float val) { labelGenerator.getMfSegmentationPerformer()->bilatSigmaLocation = val; }
void MaskFusion::setMfBilatRadius(int val) { labelGenerator.getMfSegmentationPerformer()->bilatSigmaRadius = val; }
void MaskFusion::setMfMorphEdgeRadius(int val) { labelGenerator.getMfSegmentationPerformer()->morphEdgeRadius = val; }
void MaskFusion::setMfMorphEdgeIterations(int val) { labelGenerator.getMfSegmentationPerformer()->morphEdgeIterations = val; }
void MaskFusion::setMfMorphMaskRadius(int val) { labelGenerator.getMfSegmentationPerformer()->morphMaskRadius = val; }
void MaskFusion::setMfMorphMaskIterations(int val) { labelGenerator.getMfSegmentationPerformer()->morphMaskIterations = val; }
void MaskFusion::setMfThreshold(float val) { labelGenerator.getMfSegmentationPerformer()->threshold = val; }
void MaskFusion::setMfWeightConvexity(float val) { labelGenerator.getMfSegmentationPerformer()->weightConvexity = val; }
void MaskFusion::setMfWeightDistance(float val) { labelGenerator.getMfSegmentationPerformer()->weightDistance = val; }
void MaskFusion::setTrackableClassIds(const std::set<int>& ids) { trackableClassIds = ids; }

void MaskFusion::setFernThresh(const float& val) { fernThresh = val; }

void MaskFusion::setDepthCutoff(const float& val) { depthCutoff = val; }

const bool& MaskFusion::getLost() {  // lel
    return lost;
}

const int& MaskFusion::getTick() { return tick; }

const int& MaskFusion::getTimeDelta() { return timeDelta; }

void MaskFusion::setTick(const int& val) { tick = val; }

const float& MaskFusion::getMaxDepthProcessed() { return maxDepthProcessed; }

const Eigen::Matrix4f& MaskFusion::getCurrPose() { return globalModel->getPose(); }

const int& MaskFusion::getDeforms() { return deforms; }

const int& MaskFusion::getFernDeforms() { return fernDeforms; }

std::map<std::string, FeedbackBuffer*>& MaskFusion::getFeedbackBuffers() { return feedbackBuffers; }
