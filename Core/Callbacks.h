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

#include <functional>
#include <memory>
#include <queue>
#include <mutex>
#include <thread>

template <typename Parameter>
struct CallbackBuffer {
  CallbackBuffer(char queueSize) { this->bufferSize = queueSize; }

  inline void addListener(const std::function<void(Parameter)>& listener) {
    std::lock_guard<std::mutex> lock(mutex);
    listeners.push_back(listener);
  }

  // Buffer size is checked seperately, see shrink().
  inline void addElement(const Parameter& e) {
    std::lock_guard<std::mutex> lock(mutex);
    if (buffer.size() > bufferSize) buffer.pop();
    buffer.emplace(e);
  }

  // Pass all elements to all listeners (empty buffer)
  inline void callListeners() {
    // std::lock_guard<std::mutex> lock(mutex);
    while (!buffer.empty()) {  // Fixme: Race condition
      for (auto& listener : listeners) {
        listener(buffer.front());
      }
      {
        std::lock_guard<std::mutex> lock(mutex);
        buffer.pop();
      }
    }
  }

  inline void callListenersDirect(const Parameter& e) {
    for (auto& listener : listeners) listener(e);
  }

 private:
  std::vector<std::function<void(Parameter)> > listeners;
  std::queue<Parameter> buffer;
  std::mutex mutex;
  unsigned char bufferSize;
};

class Model;
typedef std::function<void(std::shared_ptr<Model>)> ModelListener;
