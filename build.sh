#!/bin/bash -e
#
# This is a build script for MaskFusion.
#
# Use parameters:
# `--install-packages` to install required Ubuntu packages
# `--install-cuda` to install the NVIDIA CUDA suite
# `--build-dependencies` to build third party dependencies
#
# Example:
#   ./build.sh --install-packages --build-dependencies
#
#   which will create:
#   - ./deps/densecrf
#   - ./deps/gSLICr
#   - ./deps/OpenNI2
#   - ./deps/Pangolin
#   - ./deps/opencv-3.1.0
#   - ./deps/boost (unless env BOOST_ROOT is defined)
#   - ./deps/coco
#   - ./deps/Mask_RCNN

# Function that executes the clone command given as $1 iff repo does not exist yet. Otherwise pulls.
# Only works if repository path ends with '.git'
# Example: git_clone "git clone --branch 3.4.1 --depth=1 https://github.com/opencv/opencv.git"
function git_clone(){
  repo_dir=`basename "$1" .git`
  git -C "$repo_dir" pull 2> /dev/null || eval "$1"
}

# Ensure that current directory is root of project
cd $(dirname `realpath $0`)

# Enable colors
source deps/bashcolors/bash_colors.sh
function highlight(){
  clr_magentab clr_bold clr_white "$1"
}

highlight "Starting MaskFusion build script ..."
echo "Available parameters:
        --install-packages
        --install-cuda
        --build-dependencies"

if [[ $* == *--install-packages* ]] ; then
  highlight "Installing system packages..."
  # Get ubuntu version:
  sudo apt-get install -y wget software-properties-common
  source /etc/lsb-release # fetch DISTRIB_CODENAME
  if [[ $DISTRIB_CODENAME == *"trusty"* ]] ; then
    # g++ 4.9.4
    sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
    sudo apt-get install \
      g++-4.9 \
      gcc-4.9
    # cmake 3.2.2
    sudo add-apt-repository -y ppa:george-edison55/cmake-3.x
    # openjdk 8
    sudo add-apt-repository -y ppa:openjdk-r/ppa
  fi

  sudo apt-get update > /dev/null
  sudo apt-get install -y \
    build-essential \
    cmake \
    freeglut3-dev \
    git \
    g++ \
    gcc \
    libeigen3-dev \
    libglew-dev \
    libjpeg-dev \
    libsuitesparse-dev \
    libudev-dev \
    libusb-1.0-0-dev \
    openjdk-8-jdk \
    unzip \
    zlib1g-dev \
    cython3 \
    libboost-all-dev \
    libfreetype6-dev

    sudo -H pip3 install virtualenv

  if [[ $DISTRIB_CODENAME == *"trusty"* ]] ; then
     # switch to g++-4.9
     sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-4.9
     # switch to java-1.8.0
     sudo update-java-alternatives -s java-1.8.0-openjdk-amd64
  fi

fi # --install-packages


if [[ $* == *--install-cuda* ]] ; then
  highlight "Installing CUDA..."
  # Get ubuntu version:
  sudo apt-get install -y wget software-properties-common
  source /etc/lsb-release # fetch DISTRIB_CODENAME
  if [[ $DISTRIB_CODENAME == *"trusty"* ]] ; then
    # CUDA
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
    rm cuda-repo-ubuntu1404_7.5-18_amd64.deb
    sudo apt-get update > /dev/null
    sudo apt-get install -y cuda-7-5
  elif [[ $DISTRIB_CODENAME == *"vivid"* ]] ; then
    # CUDA
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1504/x86_64/cuda-repo-ubuntu1504_7.5-18_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1504_7.5-18_amd64.deb
    rm cuda-repo-ubuntu1504_7.5-18_amd64.deb
    sudo apt-get update > /dev/null
    sudo apt-get install cuda-7-5
  elif [[ $DISTRIB_CODENAME == *"xenial"* ]]; then
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
    rm cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
    sudo apt-get update > /dev/null
    sudo apt-get install -y cuda-8-0
  else
    echo "$DISTRIB_CODENAME is not yet supported"
    exit 1
  fi
fi # --install-cuda


# Create virtual python environment and install packages
highlight "Setting up virtual python environment..."
virtualenv python-environment
source python-environment/bin/activate
pip3 install pip --upgrade
pip3 install tensorflow-gpu==1.8.0
pip3 install scikit-image
pip3 install keras
pip3 install IPython
pip3 install h5py
pip3 install cython
pip3 install imgaug
pip3 install opencv-python
pip3 install pytoml
ln -s `python -c "import numpy as np; print(np.__path__[0])"`/core/include/numpy Core/Segmentation/MaskRCNN || true # Provide numpy headers to C++



if [[ $* == *--build-dependencies* ]] ; then

  # Build dependencies
  mkdir -p deps
  cd deps

  highlight "Building opencv..."
  git_clone "git clone --branch 3.4.1 --depth=1 https://github.com/opencv/opencv.git"
  cd opencv
  mkdir -p build
  cd build
  cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="`pwd`/../install" \
    \
    `# OpenCV: (building is not possible when DBUILD_opencv_video/_videoio is OFF?)` \
    -DWITH_CUDA=OFF  \
    -DBUILD_DOCS=OFF  \
    -DBUILD_PACKAGE=OFF \
    -DBUILD_TESTS=OFF  \
    -DBUILD_PERF_TESTS=OFF  \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_opencv_calib3d=OFF  \
    -DBUILD_opencv_cudaoptflow=OFF  \
    -DBUILD_opencv_dnn=OFF  \
    -DBUILD_opencv_dnn_BUILD_TORCH_IMPORTER=OFF  \
    -DBUILD_opencv_features2d=OFF \
    -DBUILD_opencv_flann=OFF \
    -DBUILD_opencv_java=OFF  \
    -DBUILD_opencv_objdetect=OFF  \
    -DBUILD_opencv_python2=OFF  \
    -DBUILD_opencv_python3=OFF  \
    -DBUILD_opencv_photo=OFF \
    -DBUILD_opencv_stitching=OFF  \
    -DBUILD_opencv_superres=OFF  \
    -DBUILD_opencv_shape=OFF  \
    -DBUILD_opencv_videostab=OFF \
    -DBUILD_PROTOBUF=OFF \
    -DWITH_1394=OFF  \
    -DWITH_GSTREAMER=OFF  \
    -DWITH_GPHOTO2=OFF  \
    -DWITH_MATLAB=OFF  \
    -DWITH_NVCUVID=OFF \
    -DWITH_OPENCL=OFF \
    -DWITH_OPENCLAMDBLAS=OFF \
    -DWITH_OPENCLAMDFFT=OFF \
    -DWITH_TIFF=OFF  \
    -DWITH_VTK=OFF  \
    -DWITH_WEBP=OFF  \
    ..
  make -j8
  cd ../build
  OpenCV_DIR=$(pwd)
  cd ../..

  if [ -z "${BOOST_ROOT}" -a ! -d boost ]; then
    highlight "Building boost..."
    wget --no-clobber -O boost_1_62_0.tar.bz2 https://sourceforge.net/projects/boost/files/boost/1.62.0/boost_1_62_0.tar.bz2/download
    tar -xjf boost_1_62_0.tar.bz2 > /dev/null
    rm boost_1_62_0.tar.bz2
    cd boost_1_62_0
    mkdir -p ../boost
    ./bootstrap.sh --prefix=../boost
    ./b2 --prefix=../boost --with-filesystem install > /dev/null
    cd ..
    rm -r boost_1_62_0
    BOOST_ROOT=$(pwd)/boost
  fi

  # build pangolin
  highlight "Building pangolin..."
  git_clone "git clone --depth=1 https://github.com/stevenlovegrove/Pangolin.git"
  cd Pangolin
  git pull
  mkdir -p build
  cd build
  cmake -DAVFORMAT_INCLUDE_DIR="" -DCPP11_NO_BOOST=ON ..
  make -j8
  Pangolin_DIR=$(pwd)
  cd ../..

  # build OpenNI2
  highlight "Building openni2..."
  git_clone "git clone --depth=1 https://github.com/occipital/OpenNI2.git"
  cd OpenNI2
  git pull
  make -j8
  cd ..

  # build freetype-gl-cpp
  highlight "Building freetype-gl-cpp..."
  git_clone "git clone --depth=1 --recurse-submodules https://github.com/martinruenz/freetype-gl-cpp.git"
  cd freetype-gl-cpp
  mkdir -p build
  cd build
  cmake -DBUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX="`pwd`/../install" -DCMAKE_BUILD_TYPE=Release ..
  make -j8
  make install
  cd ../..

  # build DenseCRF, see: http://graphics.stanford.edu/projects/drf/
  highlight "Building densecrf..."
  git_clone "git clone --depth=1 https://github.com/martinruenz/densecrf.git"
  cd densecrf
  git pull
  mkdir -p build
  cd build
  cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fPIC" \
    ..
  make -j8
  cd ../..

  # build gSLICr, see: http://www.robots.ox.ac.uk/~victor/gslicr/
  highlight "Building gslicr..."
  git_clone "git clone --depth=1 https://github.com/carlren/gSLICr.git"
  cd gSLICr
  git pull
  mkdir -p build
  cd build
  cmake \
    -DOpenCV_DIR="${OpenCV_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_HOST_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -D_FORCE_INLINES" \
    ..
  make -j8
  cd ../..

  # Prepare MaskRCNN and data
  highlight "Building mask-rcnn with ms-coco..."
  git_clone "git clone --depth=1 https://github.com/matterport/Mask_RCNN.git"
  git_clone "git clone --depth=1 https://github.com/waleedka/coco.git"
  cd coco/PythonAPI
  make
  make install # Make sure to source the correct python environment first
  cd ../..
  cd Mask_RCNN
  mkdir -p data
  cd data
  wget --no-clobber https://github.com/matterport/Mask_RCNN/releases/download/v1.0/mask_rcnn_coco.h5
  cd ../..

  # c++ toml
  highlight "Building toml11..."
  git_clone "git clone --depth=1 --branch v2.4.0 https://github.com/ToruNiina/toml11.git"

  cd ..
fi # --build-dependencies

if [ -z "${BOOST_ROOT}" -a -d deps/boost ]; then
  BOOST_ROOT=$(pwd)/deps/boost
fi

# Build MaskFusion
highlight "Building MaskFusion..."
mkdir -p build
cd build
ln -s ../deps/Mask_RCNN ./ || true # Also, make sure that the file 'mask_rcnn_model.h5' is linked or present
cmake \
  -DBOOST_ROOT="${BOOST_ROOT}" \
  -DOpenCV_DIR="$(pwd)/../deps/opencv/build" \
  -DPangolin_DIR="$(pwd)/../deps/Pangolin/build/src" \
  -DMASKFUSION_PYTHON_VE_PATH="$(pwd)/../python-environment" \
  -DCUDA_HOST_COMPILER=/usr/bin/gcc \
  -DWITH_FREENECT2=OFF \
  ..
make -j8
cd ..
