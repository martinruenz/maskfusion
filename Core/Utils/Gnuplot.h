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

#include <stdio.h>
#include <string>
#include <vector>

class GnuplotPipe {
 public:
  GnuplotPipe(bool persist = true);
  ~GnuplotPipe();

  void sendLine(const std::string& text, bool useBuffer = false);
  void sendEndOfData(unsigned repeatBuffer = 1);
  void sendNewDataBlock();

  void writeBufferToFile(const std::string& fileName);

 private:
  FILE* pipe;
  GnuplotPipe(GnuplotPipe const&) = delete;
  void operator=(GnuplotPipe const&) = delete;

  std::vector<std::string> buffer;
};
