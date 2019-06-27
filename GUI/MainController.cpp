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

#include "../Core/Utils/Macros.h"
#include "MainController.h"
#include "Tools/KlgLogReader.h"
#include "Tools/OpenNI2LiveReader.h"
#ifdef WITH_FREENECT2
#include "Tools/FreenectLiveReader.h"
#endif
#include "Tools/ImageLogReader.h"

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <GUI/Tools/PangolinReader.h>
#include <opencv2/core/ocl.hpp>
#include <toml.hpp>

/*
 * Parameters:

    -run    Run dataset immediately (otherwise start paused).
    -q      Quit when finished a log.
    -cal    Loads a camera calibration file specified as fx fy cx cy.
    -p      Loads ground truth poses to use instead of estimated pose.
    -d      Cutoff distance for depth processing (default 5m).
    -i      Relative ICP/RGB tracking weight (default 10).
    -or     Outlier rejection strength (default 3).
    -ie     Local loop closure residual threshold (default 5e-05).
    -ic     Local loop closure inlier threshold (default 35000).
    -cv     Local loop closure covariance threshold (default 1e-05).
    -pt     Global loop closure photometric threshold (default 115).
    -ft     Fern encoding threshold (default 0.3095).
    -t      Time window length (default 200).
    -s      Frames to skip at start of log.
    -e      Cut off frame of log.
    -f      Flip RGB/BGR.
    -a      Preallocate memory for a number of models, which can increase performance (default: 0)
    -icl    Enable this if using the ICL-NUIM dataset (flips normals to account for negative focal length on that data).
    -o      Open loop mode.
    -rl     Enable relocalisation.
    -fs     Frame skip if processing a log to simulate real-time.
    -fo     Fast odometry (single level pyramid).
    -nso    Disables SO(3) pre-alignment in tracking.
    -r      Rewind and loop log forever.
    -ftf    Do frame-to-frame RGB tracking.
    -sc     Showcase mode (minimal GUI).
    -vxp    Use visionx point cloud reader. Provide the name of the Provider.
    -vxf    Use together with visionx point cloud reader option. Provide the name for the file.

    -static        Disable multi-model fusion.
    -method        Method used for segmentation (cofusion, maskfusion) // FIXME, also prefix for method parameters
    -frameQ        Set size of frame-queue manually
    -confO         Initial surfel confidence threshold for objects (default 0.01).
    -confG         Initial surfel confidence threshold for scene (default 10.00).
    -segMinNew     Min size of new object segments (relative to image size)
    -segMaxNew     Max size of new object segments (relative to image size)
    -offset        Offset between creating models
    -keep          Keep all models (even bad, deactivated)

    -l             Processes a log-file (*.klg/pangolin).
    -dir           Processes a log-directory (Default: Color####.png + Depth####.exr [+ Mask####.png])
    -depthdir      Separate depth directory (==dir if not provided)
    -maskdir       Separate mask directory (==dir if not provided)
    -exportdir     Export results to this directory, otherwise not exported
    -basedir       Treat the above paths relative to this one (like depthdir = basedir + depthdir, default "")
    -colorprefix   Specify prefix of color files (=="" or =="Color" if not provided)
    -depthprefix   Specify prefix of depth files (=="" or =="Depth" if not provided)
    -maskprefix    Specify prefix of mask files (=="" or =="Mask" if not provided)
    -indexW        Number of digits of the indexes (==4 if not provided)
    -nm            Ignore Mask####.png images as soon as the provided frame was reached.
    -es            Export segmentation
    -ev            Export viewport images
    -el            Export label images
    -em            Export models (point-cloud)
    -en            Export normal images
    -ep            Export poses after finishing run (just before quitting if '-q')

    Examples:
    -basedir /mnt/path/to/my/dataset -dir color_dir -depthdir depth_dir -maskdir mask_dir -depthprefix Depth -colorprefix Color
    -maskprefix Mask -cal calibration.txt
 */

MainController::MainController(int argc, char* argv[])
    : good(true), maskFusion(0), gui(0), groundTruthOdometry(0), logReader(nullptr), usePrecomputedMasksOnly(false), framesToSkip(0), resetButton(false), resizeStream(0) {

    // Tmp variables for parameter parsing
    std::string tmpString;
    float tmpFloat;

    iclnuim = Parse::get().arg(argc, argv, "-icl", tmpString) > -1;

    std::string baseDir;
    Parse::get().arg(argc, argv, "-basedir", baseDir);
    if (baseDir.length()) baseDir += '/';

    std::string calibrationFile;
    Parse::get().arg(argc, argv, "-cal", calibrationFile);
    if (calibrationFile.size()) calibrationFile = baseDir + calibrationFile;

    // Asus is default camera (might change later)
    if(Parse::get().arg(argc, argv, "-v2", tmpString) > -1){
        Resolution::setResolution(512, 424);
        Intrinsics::setIntrinics(528, 528, 256, 212);
    } else if(Parse::get().arg(argc, argv, "-tum3", tmpString) > -1) {
        Resolution::setResolution(640, 480);
        Intrinsics::setIntrinics(535.4, 539.2, 320.1, 247.6);
    } else {
        Resolution::setResolution(640, 480);
        Intrinsics::setIntrinics(528, 528, 320, 240);
    }

    if (calibrationFile.length()) loadCalibration(calibrationFile);
    std::cout << "Calibration set to resolution: " <<
                 Resolution::getInstance().width() << "x" <<
                 Resolution::getInstance().height() <<
                 ", [fx: " << Intrinsics::getInstance().fx() <<
                 " fy: " << Intrinsics::getInstance().fy() <<
                 ", cx: " << Intrinsics::getInstance().cx() <<
                 " cy: " << Intrinsics::getInstance().cy() << "]" << std::endl;

    bool logReaderReady = false;

    Parse::get().arg(argc, argv, "-l", logFile);
    if (logFile.length()) {
        if (boost::filesystem::exists(logFile) && boost::algorithm::ends_with(logFile, ".klg")) {
            logReader = std::make_unique<KlgLogReader>(logFile, Parse::get().arg(argc, argv, "-f", tmpString) > -1);
        } else {
            logReader = std::make_unique<PangolinReader>(logFile, Parse::get().arg(argc, argv, "-f", tmpString) > -1);
        }
        logReaderReady = true;
    }

    if (!logReaderReady) {
        Parse::get().arg(argc, argv, "-dir", logFile);
        if (logFile.length()) {
            logFile += '/';  // "colorDir"
            std::string depthDir, maskDir, depthPrefix, colorPrefix, maskPrefix;
            Parse::get().arg(argc, argv, "-depthdir", depthDir);
            Parse::get().arg(argc, argv, "-maskdir", maskDir);
            Parse::get().arg(argc, argv, "-colorprefix", colorPrefix);
            Parse::get().arg(argc, argv, "-depthprefix", depthPrefix);
            Parse::get().arg(argc, argv, "-maskprefix", maskPrefix);
            if (depthDir.length()) depthDir += '/';
            //else depthDir = logFile;
            if (maskDir.length()) maskDir += '/';
            //else maskDir = logFile;
            int indexW = -1;
            ImageLogReader* imageLogReader = new ImageLogReader(baseDir + logFile, baseDir + depthDir, baseDir + maskDir,
                                                                Parse::get().arg(argc, argv, "-indexW", indexW) > -1 ? indexW : 4, colorPrefix,
                                                                depthPrefix, maskPrefix, Parse::get().arg(argc, argv, "-f", tmpString) > -1);

            // How many masks?
            int maxMasks = -1;
            if (Parse::get().arg(argc, argv, "-nm", maxMasks) > -1) {
                if (maxMasks >= 0)
                    imageLogReader->setMaxMasks(maxMasks);
                else
                    imageLogReader->ignoreMask();
            }

            logReader = std::unique_ptr<LogReader>(imageLogReader);
            usePrecomputedMasksOnly = imageLogReader->hasPrecomputedMasksOnly();
            logReaderReady = true;
        }
    }

    // Try live cameras

    // KinectV1 / Asus
    if (!logReaderReady && Parse::get().arg(argc, argv, "-v1", tmpString) > -1) {
        logReader = std::make_unique<OpenNI2LiveReader>(logFile, Parse::get().arg(argc, argv, "-f", tmpString) > -1);
        good = ((OpenNI2LiveReader*)logReader.get())->asus->ok();
    }
    // KinectV2
    if(Parse::get().arg(argc, argv, "-v2", tmpString) > -1){
#ifdef WITH_FREENECT2
        assert(!logReaderReady);
        logReader = std::make_unique<FreenectLiveReader>();
        good = ((FreenectLiveReader*)logReader.get())->isDeviceGood();
#else
        throw std::invalid_argument("v2 support not enabled, set WITH_FREENECT2 during build.");
#endif
    }

    if(!logReader){
        std::cout << "No input data" << std::endl; // Todo: Try to find camera more automatically
        exit(1);
    }

    if (logReader->hasIntrinsics() && !calibrationFile.length()) loadCalibration(logReader->getIntinsicsFile());

    if (Parse::get().arg(argc, argv, "-p", poseFile) > 0) {
        groundTruthOdometry = new GroundTruthOdometry(poseFile);
    }

    showcaseMode = Parse::get().arg(argc, argv, "-sc", tmpString) > -1;
    gui = new GUI(logFile.length() == 0, showcaseMode);

    confObjectInit = 0.01f;
    confGlobalInit = 10.0f;
    //confGlobalInit = 0.01f;
    icpErrThresh = 5e-05f;
    covThresh = 1e-05f;
    photoThresh = 115;
    fernThresh = 0.3095f;
    preallocatedModelsCount = 0;
    frameQueueSize = 30;

    timeDelta = 200;  // Ignored, since openLoop
    icpCountThresh = 40000;
    start = 1;
    so3 = !(Parse::get().arg(argc, argv, "-nso", tmpString) > -1);
    end = std::numeric_limits<unsigned short>::max();  // Funny bound, since we predict times in this format really!

    Parse::get().arg(argc, argv, "-confG", confGlobalInit);
    Parse::get().arg(argc, argv, "-confO", confObjectInit);
    Parse::get().arg(argc, argv, "-ie", icpErrThresh);
    Parse::get().arg(argc, argv, "-cv", covThresh);
    Parse::get().arg(argc, argv, "-pt", photoThresh);
    Parse::get().arg(argc, argv, "-ft", fernThresh);
    Parse::get().arg(argc, argv, "-t", timeDelta);
    Parse::get().arg(argc, argv, "-ic", icpCountThresh);
    Parse::get().arg(argc, argv, "-s", start);
    Parse::get().arg(argc, argv, "-e", end);
    Parse::get().arg(argc, argv, "-a", preallocatedModelsCount);
    Parse::get().arg(argc, argv, "-frameQ", frameQueueSize);

    logReader->flipColors = Parse::get().arg(argc, argv, "-f", tmpString) > -1;

    openLoop = true;  // FIXME //!groundTruthOdometry && (Parse::get().arg(argc, argv, "-o", empty) > -1);
    reloc = Parse::get().arg(argc, argv, "-rl", tmpString) > -1;
    frameskip = Parse::get().arg(argc, argv, "-fs", tmpString) > -1;
    quit = Parse::get().arg(argc, argv, "-q", tmpString) > -1;
    fastOdom = Parse::get().arg(argc, argv, "-fo", tmpString) > -1;
    rewind = Parse::get().arg(argc, argv, "-r", tmpString) > -1;
    frameToFrameRGB = Parse::get().arg(argc, argv, "-ftf", tmpString) > -1;
    exportSegmentation = Parse::get().arg(argc, argv, "-es", tmpString) > -1;
    exportViewport = Parse::get().arg(argc, argv, "-ev", tmpString) > -1;
    exportLabels = Parse::get().arg(argc, argv, "-el", tmpString) > -1;
    exportNormals = Parse::get().arg(argc, argv, "-en", tmpString) > -1;
    exportPoses = Parse::get().arg(argc, argv, "-ep", tmpString) > -1;
    exportModels = Parse::get().arg(argc, argv, "-em", tmpString) > -1;
    Parse::get().arg(argc, argv, "-method", tmpString);
    if (tmpString == "cofusion") {
        segmentationMethod = Segmentation::Method::CO_FUSION;
        gui->addCRFParameter();
    } else {
        segmentationMethod = Segmentation::Method::MASK_FUSION;
        gui->addBifoldParameters();
    }

    // Load configuration from files
    if(pangolin::FileExists("parameters.cfg"))
        pangolin::ParseVarsFile("parameters.cfg");
    if(!pangolin::FileExists("config.toml"))
        throw std::runtime_error("Could not read 'config.toml', which specifies class-names and weights.");

    std::vector<std::string> classNames;
    std::vector<std::string> trackableClasses;
    try {
        toml::table tomlConfig = toml::parse("config.toml");
        const auto tomlMaskRCNN = toml::get<toml::Table>(tomlConfig.at("MaskRCNN"));
        classNames = toml::get<std::vector<std::string>>(tomlMaskRCNN.at("class_names"));
        trackableClasses = toml::get<std::vector<std::string>>(tomlMaskRCNN.at("trackable_classes"));

        for(std::string& c : trackableClasses){
            trackableClassIds.insert(std::distance(classNames.begin(), std::find(classNames.begin(), classNames.end(), c)));
        }
    } catch(...){
        throw std::invalid_argument("Unable to parse input toml configuration file.");
    }

    if (Parse::get().arg(argc, argv, "-d", tmpFloat) > -1) gui->depthCutoff->Ref().Set(tmpFloat);
    if (Parse::get().arg(argc, argv, "-i", tmpFloat) > -1) gui->icpWeight->Ref().Set(tmpFloat);
    if (Parse::get().arg(argc, argv, "-or", tmpFloat) > -1) gui->outlierCoefficient->Ref().Set(tmpFloat);
    if (Parse::get().arg(argc, argv, "-segMinNew", tmpFloat) > -1) gui->minRelSizeNew->Ref().Set(tmpFloat);
    if (Parse::get().arg(argc, argv, "-segMaxNew", tmpFloat) > -1) gui->maxRelSizeNew->Ref().Set(tmpFloat);
    if (Parse::get().arg(argc, argv, "-crfRGB", tmpFloat) > -1) gui->pairwiseRGBSTD->Ref().Set(tmpFloat);
    if (Parse::get().arg(argc, argv, "-crfDepth", tmpFloat) > -1) gui->pairwiseDepthSTD->Ref().Set(tmpFloat);
    if (Parse::get().arg(argc, argv, "-crfPos", tmpFloat) > -1) gui->pairwisePosSTD->Ref().Set(tmpFloat);
    if (Parse::get().arg(argc, argv, "-crfAppearance", tmpFloat) > -1) gui->pairwiseAppearanceWeight->Ref().Set(tmpFloat);
    if (Parse::get().arg(argc, argv, "-crfSmooth", tmpFloat) > -1) gui->pairwiseSmoothnessWeight->Ref().Set(tmpFloat);
    if (Parse::get().arg(argc, argv, "-offset", tmpFloat) > -1) gui->modelSpawnOffset->Ref().Set(tmpFloat);
    if (Parse::get().arg(argc, argv, "-thNew", tmpFloat) > -1) gui->thresholdNew->Ref().Set(tmpFloat);
    if (Parse::get().arg(argc, argv, "-k", tmpFloat) > -1) gui->unaryErrorK->Ref().Set(tmpFloat);

    gui->flipColors->Ref().Set(logReader->flipColors);
    gui->rgbOnly->Ref().Set(false);
    gui->enableMultiModel->Ref().Set(Parse::get().arg(argc, argv, "-static", tmpString) <= -1);
    gui->enableSmartDelete->Ref().Set(Parse::get().arg(argc, argv, "-keep", tmpString) <= -1);
    gui->pyramid->Ref().Set(true);
    gui->fastOdom->Ref().Set(fastOdom);
    // gui->confidenceThreshold->Ref().Set(confidence);
    gui->so3->Ref().Set(so3);
    gui->frameToFrameRGB->Ref().Set(frameToFrameRGB);
    gui->pause->Ref().Set((Parse::get().arg(argc, argv, "-run", tmpString) <= -1));
    // gui->pause->Ref().Set(logFile.length());
    // gui->pause->Ref().Set(!showcaseMode);

    resizeStream = new GPUResize(Resolution::getInstance().width(), Resolution::getInstance().height(), Resolution::getInstance().width() / 2,
                                 Resolution::getInstance().height() / 2);

    if (Parse::get().arg(argc, argv, "-exportdir", exportDir) > 0) {
        if (exportDir.length() == 0 || exportDir[0] != '/') exportDir = baseDir + exportDir;
    } else {
        if (boost::filesystem::exists(logFile)) {
            // TODO: this is bound to fail if logFile is not in the baseDir or the path is not relative
            exportDir = baseDir + logFile + "-export/";
        } else {
            exportDir = baseDir + "-export/";
        }
    }
    exportDir += "/";

    // Create export dir if it doesn't exist
    boost::filesystem::path eDir(exportDir);
    boost::filesystem::create_directory(eDir);

    std::cout << "Initialised MainController. Frame resolution is set to: " << Resolution::getInstance().width() << "x"
              << Resolution::getInstance().height() << "\n"
                                                       "Exporting results to: "
              << exportDir << std::endl;

    // Setup rendering for labels
#ifdef WITH_FREETYPE_GL_CPP
    gui->addLabelTexts(classNames, {{0,0,0,1}});
#endif
}

MainController::~MainController() {
    if (maskFusion) {
        delete maskFusion;
    }

    if (gui) {
        delete gui;
    }

    if (groundTruthOdometry) {
        delete groundTruthOdometry;
    }

    if (resizeStream) {
        delete resizeStream;
    }
}

void MainController::loadCalibration(const std::string& filename) {
    std::cout << "Loading camera parameters from file: " << filename << std::endl;

    std::ifstream file(filename);
    std::string line;

    CHECK_THROW(!file.eof());

    double fx, fy, cx, cy, w, h;

    std::getline(file, line);

    int n = sscanf(line.c_str(), "%lg %lg %lg %lg %lg %lg", &fx, &fy, &cx, &cy, &w, &h);

    if (n != 4 && n != 6)
        throw std::invalid_argument("Ooops, your calibration file should contain a single line with [fx fy cx cy] or [fx fy cx cy w h]");

    Intrinsics::setIntrinics(fx, fy, cx, cy);
    if (n == 6) Resolution::setResolution(w, h);
}

void MainController::launch() {
    while (good) {
        if (maskFusion) {
            run();
        }

        if (maskFusion == 0 || resetButton) {
            resetButton = false;

            if (maskFusion) {
                delete maskFusion;
                cudaCheckError();
            }

            maskFusion = new MaskFusion(openLoop ? std::numeric_limits<int>::max() / 2 : timeDelta, icpCountThresh, icpErrThresh, covThresh,
                                    !openLoop, iclnuim, reloc, photoThresh, confGlobalInit, confObjectInit, gui->depthCutoff->Get(),
                                    gui->icpWeight->Get(), fastOdom, fernThresh, so3, frameToFrameRGB, gui->modelSpawnOffset->Get(),
                                    Model::MatchingType::Drost, segmentationMethod, exportDir, exportSegmentation, usePrecomputedMasksOnly, frameQueueSize);
            maskFusion->setTrackableClassIds(trackableClassIds);

            gui->addTextureColumn(maskFusion->getDrawableTextures());

            maskFusion->preallocateModels(preallocatedModelsCount);

            auto globalModel = maskFusion->getBackgroundModel();
            gui->addModel(globalModel->getID(), globalModel->getConfidenceThreshold());

            maskFusion->addNewModelListener(
                        [this](std::shared_ptr<Model> model) { gui->addModel(model->getID(), model->getConfidenceThreshold()); });
            // eFusion->addNewModelListener([this](std::shared_ptr<Model> model){
            //    gui->addModel(model->getID(), model->getConfidenceThreshold());}
            //);
        } else {
            break;
        }
    }
}

void MainController::run() {
    while (!pangolin::ShouldQuit() && !((!logReader->hasMore()) && quit) && !(maskFusion->getTick() == end && quit)) {
        if (!gui->pause->Get() || pangolin::Pushed(*gui->step)) {
            if ((logReader->hasMore() || rewind) && maskFusion->getTick() < end) {
                TICK("LogRead");
                if (rewind) {
                    if (!logReader->hasMore()) {
                        logReader->getPrevious();
                    } else {
                        logReader->getNext();
                    }

                    if (logReader->rewind()) {
                        logReader->currentFrame = 0;
                    }
                } else {
                    logReader->getNext();
                }
                TOCK("LogRead");

                if (maskFusion->getTick() < start) {
                    maskFusion->setTick(start);
                    logReader->fastForward(start);
                }

                float weightMultiplier = framesToSkip + 1;

                if (framesToSkip > 0) {
                    maskFusion->setTick(maskFusion->getTick() + framesToSkip);
                    logReader->fastForward(logReader->currentFrame + framesToSkip);
                    framesToSkip = 0;
                }

                Eigen::Matrix4f* currentPose = 0;

                if (groundTruthOdometry) {
                    currentPose = new Eigen::Matrix4f;
                    currentPose->setIdentity();
                    *currentPose = groundTruthOdometry->getIncrementalTransformation(logReader->getFrameData()->timestamp);
                }

                if (maskFusion->processFrame(logReader->getFrameData(), currentPose, weightMultiplier) && !showcaseMode) {
                    gui->pause->Ref().Set(true);
                }

                if (exportLabels) {
                    gui->saveColorImage(exportDir + "Labels" + std::to_string(maskFusion->getTick() - 1));
                    drawScene(DRAW_COLOR, DRAW_LABEL);
                }

                if (exportNormals) {
                    gui->saveColorImage(exportDir + "Normals" + std::to_string(maskFusion->getTick() - 1));
                    drawScene(DRAW_NORMALS, DRAW_NORMALS);
                }

                if (exportViewport) {
                    gui->saveColorImage(exportDir + "Viewport" + std::to_string(maskFusion->getTick() - 1));
                    // drawScene();
                }

                if (currentPose) {
                    delete currentPose;
                }

                if (frameskip && Stopwatch::getInstance().getTimings().at("Run") > 1000.f / 30.f) {
                    framesToSkip = int(Stopwatch::getInstance().getTimings().at("Run") / (1000.f / 30.f));
                }
            }
        } else if (pangolin::Pushed(*gui->skip)) {
            maskFusion->setTick(maskFusion->getTick() + 1);
            logReader->fastForward(logReader->currentFrame + 1);
        } else {
            //      maskFusion->predict();
            //      // TODO Only if relevant setting changed (Deactivate when writing (debug/visualisation) images to hd
            //      if (logReader->getFrameData().timestamp) maskFusion->performSegmentation(logReader->getFrameData());
        }

        TICK("GUI");

        std::stringstream stri;
        stri << maskFusion->getModelToModel().lastICPCount;
        gui->trackInliers->Ref().Set(stri.str());

        std::stringstream stre;
        stre << (std::isnan(maskFusion->getModelToModel().lastICPError) ? 0 : maskFusion->getModelToModel().lastICPError);
        gui->trackRes->Ref().Set(stre.str());

        if (!gui->pause->Get()) {
            gui->resLog.Log((std::isnan(maskFusion->getModelToModel().lastICPError) ? std::numeric_limits<float>::max()
                                                                                  : maskFusion->getModelToModel().lastICPError),
                            icpErrThresh);
            gui->inLog.Log(maskFusion->getModelToModel().lastICPCount, icpCountThresh);
        }

        drawScene();

        // GET PARAMETERS (update gui)
        ModelList& models = maskFusion->getModels();
        ModelList::iterator it = models.begin();
        for (unsigned i = 0; i < models.size(); i++) {
            ModelPointer& m = *(it++);
            gui->modelInfos[i].confThreshold->Ref().Set(m->getConfidenceThreshold());
        }

        // SET PARAMETERS / SETTINGS
        logReader->flipColors = gui->flipColors->Get();
        maskFusion->setEnableMultipleModels(gui->enableMultiModel->Get());
        maskFusion->setEnableSmartModelDelete(gui->enableSmartDelete->Get());
        maskFusion->setTrackAllModels(gui->enableTrackAll->Get());
        maskFusion->setRgbOnly(gui->rgbOnly->Get());
        maskFusion->setPyramid(gui->pyramid->Get());
        maskFusion->setFastOdom(gui->fastOdom->Get());
        maskFusion->setDepthCutoff(gui->depthCutoff->Get());
        maskFusion->setIcpWeight(gui->icpWeight->Get());
        maskFusion->setOutlierCoefficient(gui->outlierCoefficient->Get());
        maskFusion->setSo3(gui->so3->Get());
        maskFusion->setFrameToFrameRGB(gui->frameToFrameRGB->Get());

        maskFusion->setModelSpawnOffset(gui->modelSpawnOffset->Get());
        maskFusion->setModelDeactivateCount(gui->modelDeactivateCnt->Get());
        maskFusion->setNewModelMinRelativeSize(gui->minRelSizeNew->Get());
        maskFusion->setNewModelMaxRelativeSize(gui->maxRelSizeNew->Get());

        if (segmentationMethod == Segmentation::Method::CO_FUSION) {
            maskFusion->setCfPairwiseWeightAppearance(gui->pairwiseAppearanceWeight->Get());
            maskFusion->setCfPairwiseWeightSmoothness(gui->pairwiseSmoothnessWeight->Get());
            maskFusion->setCfPairwiseSigmaDepth(gui->pairwiseDepthSTD->Get());
            maskFusion->setCfPairwiseSigmaPosition(gui->pairwisePosSTD->Get());
            maskFusion->setCfPairwiseSigmaRGB(gui->pairwiseRGBSTD->Get());
            maskFusion->setCfThresholdNew(gui->thresholdNew->Get());
            maskFusion->setCfUnaryKError(gui->unaryErrorK->Get());
            maskFusion->setCfUnaryWeightError(gui->unaryErrorWeight->Get());
            maskFusion->setCfIteration(gui->crfIterations->Get());
        } else if (segmentationMethod == Segmentation::Method::MASK_FUSION) {
            maskFusion->setMfBilatSigmaDepth(gui->bifoldBilateralSigmaDepth->Get());
            maskFusion->setMfBilatSigmaColor(gui->bifoldBilateralSigmaColor->Get());
            maskFusion->setMfBilatSigmaLocation(gui->bifoldBilateralSigmaLocation->Get());
            maskFusion->setMfBilatRadius(gui->bifoldBilateralRadius->Get());

            maskFusion->setMfMorphEdgeRadius(gui->bifoldMorphEdgeRadius->Get());
            maskFusion->setMfMorphEdgeIterations(gui->bifoldMorphEdgeIterations->Get());
            maskFusion->setMfMorphMaskRadius(gui->bifoldMorphMaskRadius->Get());
            maskFusion->setMfMorphMaskIterations(gui->bifoldMorphMaskIterations->Get());

            maskFusion->setMfThreshold(gui->bifoldEdgeThreshold->Get());
            maskFusion->setMfWeightConvexity(gui->bifoldWeightConvexity->Get());
            maskFusion->setMfWeightDistance(gui->bifoldWeightDistance->Get());
            maskFusion->setMfNonstaticThreshold(gui->bifoldNonstaticThreshold->Get());
        }

        resetButton = pangolin::Pushed(*gui->reset);

        if (gui->autoSettings) {
            static bool last = gui->autoSettings->Get();

            if (gui->autoSettings->Get() != last) {
                last = gui->autoSettings->Get();
                // static_cast<LiveLogReader *>(logReader)->setAuto(last);
                logReader->setAuto(last);
            }
        }

        Stopwatch::getInstance().sendAll();

        if (resetButton) {
            break;
        }

        if (pangolin::Pushed(*gui->saveCloud)) maskFusion->savePly();
        // if(pangolin::Pushed(*gui->saveDepth)) eFusion->saveDepth();
        if (pangolin::Pushed(*gui->savePoses)) maskFusion->exportPoses();
        if (pangolin::Pushed(*gui->saveView)) {
            static int index = 0;
            std::string viewPath;
            do {
                viewPath = exportDir + "/view" + std::to_string(index++);
            } while (boost::filesystem::exists(viewPath + ".png"));
            gui->saveColorImage(viewPath);
        }

        TOCK("GUI");
    }
    if (exportPoses) maskFusion->exportPoses();
    if (exportModels) maskFusion->savePly();
}

void MainController::drawScene(DRAW_COLOR_TYPE backgroundColor, DRAW_COLOR_TYPE objectColor) {
    if (gui->followPose->Get()) {
        pangolin::OpenGlMatrix mv;

        Eigen::Matrix4f currPose = maskFusion->getCurrPose();
        Eigen::Matrix3f currRot = currPose.topLeftCorner(3, 3);

        Eigen::Quaternionf currQuat(currRot);
        Eigen::Vector3f forwardVector(0, 0, 1);
        Eigen::Vector3f upVector(0, iclnuim ? 1 : -1, 0);

        Eigen::Vector3f forward = (currQuat * forwardVector).normalized();
        Eigen::Vector3f up = (currQuat * upVector).normalized();

        Eigen::Vector3f eye(currPose(0, 3), currPose(1, 3), currPose(2, 3));

        const float shift = 0.2;
        eye -= shift*forward;

        Eigen::Vector3f at = eye + forward;

        Eigen::Vector3f z = (eye - at).normalized();   // Forward
        Eigen::Vector3f x = up.cross(z).normalized();  // Right
        Eigen::Vector3f y = z.cross(x);

        Eigen::Matrix4d m;
        m << x(0), x(1), x(2), -(x.dot(eye)), y(0), y(1), y(2), -(y.dot(eye)), z(0), z(1), z(2), -(z.dot(eye)), 0, 0, 0, 1;

        memcpy(&mv.m[0], m.data(), sizeof(Eigen::Matrix4d));

        gui->s_cam.SetModelViewMatrix(mv);
    }

    gui->preCall();

    Eigen::Matrix4f pose = maskFusion->getCurrPose();
    Eigen::Matrix4f viewprojection =
            Eigen::Map<Eigen::Matrix<pangolin::GLprecision, 4, 4>>(gui->s_cam.GetProjectionModelViewMatrix().m).cast<float>();

    if (gui->drawRawCloud->Get() || gui->drawFilteredCloud->Get()) {
        maskFusion->computeFeedbackBuffers();
    }

    if (gui->drawRawCloud->Get()) {
        maskFusion->getFeedbackBuffers()
                .at(FeedbackBuffer::RAW)
                ->render(gui->s_cam.GetProjectionModelViewMatrix(), pose, gui->drawNormals->Get(), gui->drawColors->Get());
    }

    if (gui->drawFilteredCloud->Get()) {
        maskFusion->getFeedbackBuffers()
                .at(FeedbackBuffer::FILTERED)
                ->render(gui->s_cam.GetProjectionModelViewMatrix(), pose, gui->drawNormals->Get(), gui->drawColors->Get());
    }

    TICK("RENDER");

    // Compute tmp per object data
    struct ObjectRenderData {
        const ModelPointer& mptr;
        const Eigen::AlignedBox3f& bb;
        Eigen::Matrix4f ow; // obj -> world
        inline Eigen::Vector3f toGlobal(const Eigen::Vector4f& v) const {
            return (ow * v).head(3);
        }
        inline Eigen::Vector3f toGlobal(const Eigen::Vector3f& v) const {
            return toGlobal(Eigen::Vector4f(v(0),v(1),v(2),1));
        }
    };
    std::vector<ObjectRenderData> renderData;
    renderData.reserve(maskFusion->getModels().size()-1);
    auto itBegin = maskFusion->getModels().begin();
    auto itEnd = maskFusion->getModels().end();
    ModelPointer globalModel = *itBegin++;  // Skip global
    for (auto model = itBegin; model != itEnd; model++) {
        renderData.push_back({*model,
                              (*model)->getBoundingBox(),
                              pose * (*model)->getPose().inverse()});
    }

    if (false) {
        glFinish();
        gui->drawFXAA(viewprojection,  // gui->s_cam.GetProjectionModelViewMatrix(),
                      pose, gui->s_cam.GetModelViewMatrix(), maskFusion->getModels(), maskFusion->getTick(), maskFusion->getTimeDelta(), iclnuim);

        glFinish();
    } else {
        int selectedColorType =
                gui->drawNormals->Get() ? 1 : gui->drawColors->Get() ? 2 : gui->drawTimes->Get() ? 3 : gui->drawLabelColors->Get() ? 4 : 2;
        int globalColorType = selectedColorType;
        int objectColorType = selectedColorType;
        if (backgroundColor != DRAW_USER_DEFINED) globalColorType = backgroundColor;
        if (objectColor != DRAW_USER_DEFINED) objectColorType = objectColor;

        if (gui->drawGlobalModel->Get()) {
            maskFusion->getBackgroundModel()->renderPointCloud(viewprojection, gui->drawUnstable->Get(), gui->drawPoints->Get(),
                                                             gui->drawWindow->Get(), globalColorType, maskFusion->getTick(),
                                                             maskFusion->getTimeDelta());
        }


        if (gui->drawObjectModels->Get()) {
            // globalModel->bindRenderPointCloudShader(gui->drawPoints->Get()); // TODO enable this (bind program only once!)
            glColor3f(1,0,0); // Todo Change color of bounding box?
            for (size_t i=0; i < renderData.size(); i++) {
                const ObjectRenderData& oData = renderData[i];
                oData.mptr->renderPointCloud(viewprojection * oData.ow, gui->drawUnstable->Get(), gui->drawPoints->Get(),
                                           gui->drawWindow->Get(), objectColorType, maskFusion->getTick(), maskFusion->getTimeDelta());


                if (gui->drawBoundingBoxes->Get() && oData.bb.min()(2) > -6 && oData.bb.min()(2) < 6) {
                    glMatrixMode(GL_MODELVIEW);
                    glPushMatrix();
                    glMultMatrixf(oData.ow.data());
                    if(oData.mptr->isNonstatic()){
                        glLineWidth(2);
                        glColor3f(0,0,1);
                        pangolin::glDrawAlignedBox(oData.bb);
                        glLineWidth(1);
                        glColor3f(1,0,0);
                    } else {
                        glLineWidth(2);
                        pangolin::glDrawAlignedBox(oData.bb);
                    }
                    glPopMatrix();
                }
            }
        }
        glFlush();
        //globalModel->unbindRenderPointCloudShader(gui->drawPoints->Get());

    }

    if(renderData.size()>2 && false){
        // Draw sphere at cam
        glPushMatrix();
        glMultMatrixf(pose.data());
        glColor3f(1, 0, 0);
        drawSphere(0.07, 16, 16);
        glPopMatrix();

        // Connect cam / first object
        ObjectRenderData& rd = renderData[2];
        Eigen::Vector3f p = rd.toGlobal(Eigen::Vector3f(rd.bb.center()));
        Eigen::Vector3f cp = pose.topRightCorner(3,1);
        drawLineFromTo(cp,p,8);
    }

    // Export model info (bb, center ...)
    if(false){
        static int last_tick = -1;
        int tick = maskFusion->getTick();

        if(tick != last_tick){
            for (size_t i=0; i < renderData.size(); i++) {
                std::ofstream outfile;
                const ObjectRenderData& rd = renderData[i];
                Eigen::Vector3f minb = rd.toGlobal(rd.bb.min());
                Eigen::Vector3f maxb = rd.toGlobal(rd.bb.max());
                Eigen::Vector3f centerb = rd.toGlobal(Eigen::Vector3f(rd.bb.center()));
                Eigen::Vector3f x = rd.toGlobal(Eigen::Vector4f(1,0,0,0));
                Eigen::Vector3f y = rd.toGlobal(Eigen::Vector4f(0,1,0,0));
                Eigen::Vector3f z = rd.toGlobal(Eigen::Vector4f(0,0,1,0));
                outfile.open(std::string("/tmp/bb-up-") + std::to_string(rd.mptr->getID())+".txt", std::ios_base::app);
                outfile << maskFusion->getTick() << " " <<
                           minb(0) << " " << minb(1) << " " << minb(2) << " " <<
                           maxb(0) << " " << maxb(1) << " " << maxb(2) << " " <<
                           centerb(0) << " " << centerb(1) << " " << centerb(2) << " " <<
                           x(0) << " " << x(1) << " " << x(2) << " " <<
                           y(0) << " " << y(1) << " " << y(2) << " " <<
                           z(0) << " " << z(1) << " " << z(2) << "\n";
                outfile.close();
            }
        }
        last_tick = tick;
    }

    TOCK("RENDER");

#ifdef WITH_FREETYPE_GL_CPP
    TICK("LABEL-RENDERING");
    gui->textRenderer.setProjection(gui->s_cam.GetProjectionMatrix().operator Eigen::Matrix<float, 4,4>());
    gui->textRenderer.setView(gui->s_cam.GetModelViewMatrix().operator Eigen::Matrix<float, 4,4>());
    gui->textRenderer.preRender();
    for (size_t i=0; i < renderData.size(); i++) {
        const ObjectRenderData& oData = renderData[i];
        FreetypeGlText& label = gui->textLabels[oData.mptr->getClassID()].first;
        Eigen::Affine3f t(Eigen::Translation3f(oData.bb.min()(0),oData.bb.min()(1),oData.bb.min()(2)) * Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX()));
        label.setPose(oData.ow * t.matrix());
        gui->textRenderer.renderText(label, false);
    }
    gui->textRenderer.postRender();
    TOCK("LABEL-RENDERING");
#endif

    if (gui->drawPoseLog->Get()) {
        bool object = false;
        for (auto& model : maskFusion->getModels()) {
            const std::vector<Model::PoseLogItem>& poseLog = model->getPoseLog();

            glColor3f(0, 1, 1);
            glBegin(GL_LINE_STRIP);
            for (const auto& item : poseLog) {
                glVertex3f(item.p(0), item.p(1), item.p(2));
            }
            glEnd();
            if (object) {
                glColor3f(0, 1, 0.2);
                gui->drawFrustum(pose * model->getPose().inverse());
                glColor3f(1, 1, 0.2);
            }
            object = true;
        }
    }

    const bool drawCamera = false;
    if (drawCamera) {
        maskFusion->getLost() ? glColor3f(1, 1, 0) : glColor3f(1, 0, 1);
        glLineWidth(2);
        gui->drawFrustum(pose);

        // Draw axis
        if(false){
            glPushMatrix();
            glMultMatrixf(pose.data());
            pangolin::glDrawAxis(0.05);
            glPopMatrix();
        }
    }

    if(false){
        pangolin::glDrawAxis(1);
    }

    glColor3f(1, 1, 1);


    if (gui->drawFerns->Get()) {
        glColor3f(0, 0, 0);
        for (size_t i = 0; i < maskFusion->getFerns().frames.size(); i++) {
            if ((int)i == maskFusion->getFerns().lastClosest) continue;

            gui->drawFrustum(maskFusion->getFerns().frames.at(i)->pose);
        }
        glColor3f(1, 1, 1);
    }

    if (gui->drawDefGraph->Get()) {
        const std::vector<GraphNode*>& graph = maskFusion->getLocalDeformation().getGraph();

        for (size_t i = 0; i < graph.size(); i++) {
            pangolin::glDrawCross(graph.at(i)->position(0), graph.at(i)->position(1), graph.at(i)->position(2), 0.1);

            for (size_t j = 0; j < graph.at(i)->neighbours.size(); j++) {
                pangolin::glDrawLine(graph.at(i)->position(0), graph.at(i)->position(1), graph.at(i)->position(2),
                                     graph.at(graph.at(i)->neighbours.at(j))->position(0), graph.at(graph.at(i)->neighbours.at(j))->position(1),
                                     graph.at(graph.at(i)->neighbours.at(j))->position(2));
            }
        }
    }

    if (maskFusion->getFerns().lastClosest != -1) {
        glColor3f(1, 0, 0);
        gui->drawFrustum(maskFusion->getFerns().frames.at(maskFusion->getFerns().lastClosest)->pose);
        glColor3f(1, 1, 1);
    }

    const std::vector<PoseMatch>& poseMatches = maskFusion->getPoseMatches();

    int maxDiff = 0;
    for (size_t i = 0; i < poseMatches.size(); i++) {
        if (poseMatches.at(i).secondId - poseMatches.at(i).firstId > maxDiff) {
            maxDiff = poseMatches.at(i).secondId - poseMatches.at(i).firstId;
        }
    }

    for (size_t i = 0; i < poseMatches.size(); i++) {
        if (gui->drawDeforms->Get()) {
            if (poseMatches.at(i).fern) {
                glColor3f(1, 0, 0);
            } else {
                glColor3f(0, 1, 0);
            }
            for (size_t j = 0; j < poseMatches.at(i).constraints.size(); j++) {
                pangolin::glDrawLine(poseMatches.at(i).constraints.at(j).sourcePoint(0), poseMatches.at(i).constraints.at(j).sourcePoint(1),
                                     poseMatches.at(i).constraints.at(j).sourcePoint(2), poseMatches.at(i).constraints.at(j).targetPoint(0),
                                     poseMatches.at(i).constraints.at(j).targetPoint(1), poseMatches.at(i).constraints.at(j).targetPoint(2));
            }
        }
    }
    glColor3f(1, 1, 1);

    if (!showcaseMode) {
        // Generate textures, which are specifically for visualisation
        maskFusion->normaliseDepth(0.3f, gui->depthCutoff->Get());
        maskFusion->coloriseMasks();

        // Render textures to viewports
        for (auto& name_texture : maskFusion->getDrawableTextures())
            if (name_texture.second->draw)
                gui->displayImg(name_texture.first, name_texture.second.get());

        auto itBegin = maskFusion->getModels().begin();
        auto itEnd = maskFusion->getModels().end();
        int i = 0;
        for (auto model = itBegin; model != itEnd; model++) {
            gui->displayImg("ICP" + std::to_string(++i), (*model)->getICPErrorTexture());
            if (i >= 4) break;
        }
        for (; i < 4;) {
            gui->displayEmpty("ICP" + std::to_string(++i));
            gui->displayEmpty("P" + std::to_string(i));
        }
    }

    std::stringstream strs;
    strs << maskFusion->getBackgroundModel()->lastCount();

    gui->totalPoints->operator=(strs.str());

    std::stringstream strs2;
    strs2 << maskFusion->getLocalDeformation().getGraph().size();

    gui->totalNodes->operator=(strs2.str());

    std::stringstream strs3;
    strs3 << maskFusion->getFerns().frames.size();

    gui->totalFerns->operator=(strs3.str());

    std::stringstream strs4;
    strs4 << maskFusion->getDeforms();

    gui->totalDefs->operator=(strs4.str());

    std::stringstream strs5;
    strs5 << maskFusion->getTick() << "/" << logReader->getNumFrames();

    gui->logProgress->operator=(strs5.str());

    std::stringstream strs6;
    strs6 << maskFusion->getFernDeforms();

    gui->totalFernDefs->operator=(strs6.str());

    gui->postCall();
}

void MainController::drawSphere(const float radius, const unsigned latLines, const unsigned longLines){
    const float plat_step = 1.0f / latLines;
    const float plong_step = 1.0f / longLines;

    for(unsigned i = 1; i <= latLines; i++) {
        float plat = i * plat_step;
        float lat0 = M_PI * (-0.5 + (i - 1) * plat_step); // -0.7, -0.5, -0.3, -0.1, 0.1, 0.3  *PI
        float o00  = sin(lat0);
        float o01 =  cos(lat0);

        float lat1 = M_PI * (-0.5 + plat); // [-0.5pi..0.5pi]
        float o10 = sin(lat1);
        float o11 = cos(lat1);

        glBegin(GL_QUAD_STRIP);
        for(unsigned j = 0; j <= longLines; j++) {
            float p = j * plong_step;
            float lng = 2 * M_PI * p; // [0..2pi]
            float x = -radius * sin(lng);
            float z = radius * cos(lng);
            glVertex3f(x * o01, radius * o00, z * o01);
            glVertex3f(x * o11, radius * o10, z * o11);
        }
        glEnd();
    }
}

void MainController::drawLineFromTo(const Eigen::Vector3f &p0, const Eigen::Vector3f &p1, float lineWidth){
    glLineWidth(lineWidth);
    glBegin(GL_LINES);
    glVertex3f(p1(0),p1(1),p1(2));
    glVertex3f(p0(0),p0(1),p0(2));
    glEnd();
}
