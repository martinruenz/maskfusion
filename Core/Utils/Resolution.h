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

#ifndef RESOLUTION_H_
#define RESOLUTION_H_

#include <cassert>

class Resolution {
 public:
  static const Resolution& getInstance() { return getInstancePrivate(); }

  static void setResolution(int width, int height) {
    getInstancePrivate().imgWidth = width;
    getInstancePrivate().imgHeight = height;
    getInstancePrivate().imgNumPixels = width * height;
  }

  const int& width() const {
    checkSet();
    return imgWidth;
  }

  const int& height() const {
    checkSet();
    return imgHeight;
  }

  const int& cols() const {
    checkSet();
    return imgWidth;
  }

  const int& rows() const {
    checkSet();
    return imgHeight;
  }

  const int& numPixels() const {
    checkSet();
    return imgNumPixels;
  }

 private:
  Resolution() {}
  static Resolution& getInstancePrivate() {
    static Resolution instance;
    return instance;
  }

  void checkSet() const { assert(imgWidth > 0 && imgHeight > 0 && "You haven't initialised the Resolution class!"); }

  int imgWidth = 0;
  int imgHeight = 0;
  int imgNumPixels = 0;
};

#endif /* RESOLUTION_H_ */
