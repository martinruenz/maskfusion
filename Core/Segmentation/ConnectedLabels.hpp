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

#include <opencv2/imgproc/imgproc.hpp>
#include <map>

struct ComponentData {
  unsigned char label;
  int top = std::numeric_limits<int>::max();
  int right = 0;
  int bottom = 0;
  int left = std::numeric_limits<int>::max();
  float centerX = 0;
  float centerY = 0;
  int size = 0;
};

// This should not be faster than the next one, which is easier to use.
// void mapLabelsToComponents(const std::vector<ComponentData>& ccStats, std::map<int, std::list<int>>& labelToComponents){
//    assert(labelToComponents.find(ccStats[i].label) != labelToComponents.end());
//    for(unsigned i=0; i < ccStats.size(); i++) labelToComponents[ccStats[i].label].push_back(i);
//}

std::map<int, std::list<int>> mapLabelsToComponents(const std::vector<ComponentData>& ccStats) {
  std::map<int, std::list<int>> labelToComponents;
  for (unsigned i = 0; i < ccStats.size(); i++) {
    auto r = labelToComponents.insert({ccStats[i].label, std::list<int>()});
    r.first->second.push_back(i);
  }
  // Don't worry, return-value-optimization is your friend
  return labelToComponents;
}

cv::Mat connectedLabels(cv::Mat input, std::vector<ComponentData>* stats) {
  assert(input.type() == CV_8UC1);

  cv::Mat componentsImg(input.rows, input.cols, cv::DataType<int>::type);
  int* componentRowPtr = componentsImg.ptr<int>(0);

  std::vector<int> componentRoots;
  auto newComponent = [&componentRoots]() -> int {
    int r = componentRoots.size();
    componentRoots.push_back(r);
    return r;
  };
  auto findRoot = [&componentRoots](int index) -> int {
    while (true) {
      if (index == componentRoots[index]) return index;
      index = componentRoots[index];
    }
  };
  auto merge = [&](int index1, int index2) -> int {
    int r1 = findRoot(index1);
    int r2 = findRoot(index2);
    if (r1 < r2) {
      componentRoots[r2] = r1;
      return r1;
    } else {
      componentRoots[r1] = r2;
      return r2;
    }
  };

  // First pass

  // First row
  componentRowPtr[0] = newComponent();
  for (int c = 1; c < input.cols; c++) {
    if (input.data[c] == input.data[c - 1])
      componentRowPtr[c] = componentRowPtr[c - 1];
    else
      componentRowPtr[c] = newComponent();
  }

  // Other rows
  uchar* lastRowPtr = input.ptr<uchar>(0);
  int* lastComponentRowPtr = componentRowPtr;
  for (int r = 1; r < input.rows; r++) {
    uchar* rowPtr = input.ptr<uchar>(r);
    componentRowPtr = componentsImg.ptr<int>(r);
    // First column
    if (rowPtr[0] == lastRowPtr[0])
      componentRowPtr[0] = lastComponentRowPtr[0];
    else
      componentRowPtr[0] = newComponent();
    // Other columns
    for (int c = 1; c < input.cols; c++) {
      // const uchar val = rowPtr[c];
      if (rowPtr[c] == rowPtr[c - 1]) {
        int cLeft = componentRowPtr[c - 1];
        int cTop = lastComponentRowPtr[c];
        if (rowPtr[c] == lastRowPtr[c] && cLeft != cTop) {
          // Found merging situation
          componentRowPtr[c] = merge(cTop, cLeft);
        } else {
          componentRowPtr[c] = cLeft;
        }
      } else if (rowPtr[c] == lastRowPtr[c]) {
        componentRowPtr[c] = lastComponentRowPtr[c];
      } else {
        componentRowPtr[c] = newComponent();
      }
    }
    lastRowPtr = rowPtr;
    lastComponentRowPtr = componentRowPtr;
  }

  // Second pass

  // std::map<int,int> rootMapping; // A map would save memory but is probably slower
  std::vector<int> rootMapping(componentRoots.size());
  int rootCnt = 0;
  for (unsigned id = 0; id < componentRoots.size(); id++) {
    int root = findRoot(id);
    if ((unsigned)root == id) {
      rootMapping[root] = rootCnt++;
    } else {
      componentRoots[id] = root;
    }
  }
  for (auto& c : componentRoots) c = rootMapping[c];

  // Apply second pass and optionally compute some stats
  if (stats) {
    assert(stats->size() == 0);
    stats->resize(rootCnt);
    for (int y = 0; y < componentsImg.rows; y++) {
      componentRowPtr = componentsImg.ptr<int>(y);
      uchar* rowPtr = input.ptr<uchar>(y);
      for (int x = 0; x < componentsImg.cols; x++) {
        int c = componentRoots[componentRowPtr[x]];
        componentRowPtr[x] = c;
        ComponentData& data = (*stats)[c];
        data.size++;
        data.label = rowPtr[x];
        data.centerX += x;
        data.centerY += y;
        if (y < data.top) data.top = y;
        if (y > data.bottom) data.bottom = y;
        if (x < data.left) data.left = x;
        if (x > data.right) data.right = x;
      }
    }
    for (ComponentData& data : *stats) {
      data.centerX /= data.size;
      data.centerY /= data.size;
    }
  } else {
    int* componentPtr = (int*)componentsImg.data;
    for (unsigned i = 0; i < componentsImg.total(); i++) {
      componentPtr[i] = componentRoots[componentPtr[i]];
    }
  }

  return componentsImg;
}
