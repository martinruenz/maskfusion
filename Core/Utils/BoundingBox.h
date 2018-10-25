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

// Todo make template
struct BoundingBox {
    int top = std::numeric_limits<int>::max(); // order matters
    int right = std::numeric_limits<int>::min();
    int bottom = std::numeric_limits<int>::min();
    int left = std::numeric_limits<int>::max();

    inline int w() const { return right-left; }
    inline int h() const { return bottom-top; }

    static inline BoundingBox fromLeftTopWidthHeight(int l, int t, int w, int h){
        return BoundingBox({t,l+w,t+h,l});
    }
    inline void merge(const BoundingBox& other) {
        if (other.left < left) left = other.left;
        if (other.top < top) top = other.top;
        if (other.right > right) right = other.right;
        if (other.bottom > bottom) bottom = other.bottom;
    }
    inline void mergeLeftTopWidthHeight(int l, int t, int w, int h){
        merge(fromLeftTopWidthHeight(l,t,w,h));
    }
    inline bool includes(const BoundingBox& other) const {
        return (other.left > left && other.right < right && other.top > top && other.bottom < bottom);
    }
    inline bool includesLeftTopWidthHeight(int l, int t, int w, int h) const {
        return includes(BoundingBox({t,l+w,t+h,l}));
    }
    inline void include(int y, int x) {
        top = std::min(y, top);
        right = std::max(x, right);
        bottom = std::max(y, bottom);
        left = std::min(x, left);
    }
    inline BoundingBox extended(int border_size) const {
        return BoundingBox({top-border_size, right+border_size, bottom+border_size, left-border_size});
    }
    inline BoundingBox intersection(const BoundingBox& other) const {
        return BoundingBox({std::max(top,other.top), std::min(right,other.right), std::min(bottom,other.bottom), std::max(left,other.left)});
    }
    inline bool intersects(const BoundingBox& other) const {
        return intersection(other).isPositive();
    }
    inline bool isPositive() const {
        return top <= bottom && left <= right;
    }

    // OpenCV interface
    inline cv::Rect toCvRect() const {
        return cv::Rect(left,top,w(),h());
    }
    inline void draw(cv::Mat img, cv::Scalar color=cv::Scalar(255,0,0), int thickness=1) const {
        cv::rectangle(img, toCvRect(), color, thickness);
    }

};
