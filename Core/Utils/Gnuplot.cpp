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

#include "Gnuplot.h"
#include <iostream>
#include <fstream>

GnuplotPipe::GnuplotPipe(bool persist) {
  std::cout << "Opening gnuplot... ";
  pipe = popen(persist ? "gnuplot -persist" : "gnuplot", "w");
  if (!pipe)
    std::cout << "failed!" << std::endl;
  else
    std::cout << "succeded." << std::endl;
}

GnuplotPipe::~GnuplotPipe() {
  if (pipe) pclose(pipe);
}

void GnuplotPipe::sendLine(const std::string& text, bool useBuffer) {
  if (!pipe) return;
  if (useBuffer)
    buffer.push_back(text + "\n");
  else
    fputs((text + "\n").c_str(), pipe);
}

void GnuplotPipe::sendEndOfData(unsigned repeatBuffer) {
  if (!pipe) return;
  for (unsigned i = 0; i < repeatBuffer; i++) {
    for (auto& line : buffer) fputs(line.c_str(), pipe);
    fputs("e\n", pipe);
  }
  fflush(pipe);
  buffer.clear();
}

void GnuplotPipe::sendNewDataBlock() { sendLine("\n", !buffer.empty()); }

void GnuplotPipe::writeBufferToFile(const std::string& fileName) {
  std::ofstream fileOut(fileName);
  for (auto& line : buffer) fileOut << line;
  fileOut.close();
}
