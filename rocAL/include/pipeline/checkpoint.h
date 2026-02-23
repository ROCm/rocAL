/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
#include <stdexcept>
#include <string>
#include <vector>

/*!
 * \file
 * \brief Checkpointing helpers for capturing operator state and RNG state.
 */

/*! \brief Serialize an mt19937 RNG state into a string for checkpointing. */
inline std::string SerializeRNGToString(const std::mt19937 &rng) {
    std::stringstream stream;
    stream << rng;
    return stream.str();
}

/*! \brief Holds per-operator checkpoint state during serialization. */
class OperatorCheckpoint {
   public:
    explicit OperatorCheckpoint(std::string name) : _operator_name(std::move(name)) {}

    /*! \brief Return mutable storage for the operator-specific state. */
    std::any &GetMutableCheckpointState() {
        return _state;
    }

    /*! \brief Return the typed operator checkpoint state. */
    template <typename T>
    const T &GetOperatorCheckpointState() const {
        return std::any_cast<const T &>(_state);
    }

   private:
    const std::string _operator_name;  //!< Operator name associated with this checkpoint entry.
    std::any _state;                   //!< Operator-specific checkpoint payload.
};

/*! \brief Aggregates per-operator checkpoints for a single pipeline iteration. */
class Checkpoint {
   public:
    /*! \brief Clear all stored operator checkpoints. */
    void Clear() {
        _op_cpts.clear();
        _name_to_id.clear();
    }

    /*! \brief Add a checkpoint entry for an operator and return it. */
    std::shared_ptr<OperatorCheckpoint> AddOperatorCheckpoint(std::string op_name) {
        _name_to_id[op_name] = _op_cpts.size();
        _op_cpts.emplace_back(std::make_shared<OperatorCheckpoint>(std::move(op_name)));
        return _op_cpts.back();
    }

    /*! \brief Return the checkpoint entry for a given operator name. */
    const std::shared_ptr<OperatorCheckpoint> &GetOperatorCheckpoint(const std::string &op_name) {
        auto it = _name_to_id.find(op_name);
        if (it == _name_to_id.end() || it->second >= _op_cpts.size()) {
            throw std::out_of_range("Operator checkpoint not found: " + op_name);
        }
        return _op_cpts[it->second];
    }

   private:
    std::vector<std::shared_ptr<OperatorCheckpoint>> _op_cpts;  //!< Ordered list of operator checkpoints.
    std::map<std::string, size_t, std::less<>> _name_to_id;     //!< Operator name to checkpoint index map.
};
