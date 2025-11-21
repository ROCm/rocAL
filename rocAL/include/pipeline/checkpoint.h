/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <any>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>

inline std::string SerializeRNGToString(const std::mt19937 &rng) {
    std::stringstream stream;
    stream << rng;
    return stream.str();
}

inline void DeserializeRNGFromString(const std::string &data, std::mt19937 &rng) {
    std::stringstream stream(data);
    stream >> rng;
}

class OperatorCheckpoint {
   public:
    explicit OperatorCheckpoint(std::string name) : _operator_name(std::move(name)) {}

    std::any &GetMutableCheckpointState() {
        return _state;
    }

    template <typename T>
    const T &GetOperatorCheckpointState() const {
        return std::any_cast<const T &>(_state);
    }

   private:
    const std::string _operator_name;
    std::any _state;
};

class Checkpoint {
   public:
    std::shared_ptr<OperatorCheckpoint> AddOperatorCheckpoint(std::string op_name) {
        _name_to_id[op_name] = _op_cpts.size();
        _op_cpts.emplace_back(std::make_shared<OperatorCheckpoint>(std::move(op_name)));
        return _op_cpts.back();
    }

    const std::shared_ptr<OperatorCheckpoint> &GetOperatorCheckpoint(const std::string &op_name) {
        return _op_cpts[_name_to_id[op_name]];
    }

   private:
    std::vector<std::shared_ptr<OperatorCheckpoint>> _op_cpts;
    std::map<std::string, size_t, std::less<>> _name_to_id;
};
