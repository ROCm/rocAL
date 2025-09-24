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
#include <memory>
#include <any>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>
#include <stdexcept>

#include "pipeline/argument_types.h"
#include "pipeline/enum_registry.h"
#include "parameters/parameter_factory.h"
#include "pipeline/commons.h"

/**
 * @brief Argument class stores the details of each argument in the Node
 * 
 * This class encapsulates argument information for pipeline nodes, supporting
 * various data types including basic types, enums, vectors, maps, and parameters. 
 */
class Argument {
   public:
    std::string arg_name;       ///< Name of the argument
    std::string type_name;      ///< Denotes the data type of the argument
    std::string enum_type_name; ///< Denotes the name of the enum <arg_name_enum>
    bool is_vector = false;     ///< True if the argument contains vector data
    bool is_parameter = false;  ///< True if the argument is a parameter object
    bool is_null_ptr = false;   ///< True if the argument represents a null pointer
    std::vector<std::any> values; ///< Storage for argument values (can change to std::variant later)
    pParam param;      ///< Parameter stored for parameter-type arguments
    
   private:
    // Helper method to get type name from registry or built-in types
    template<typename T>
    std::string getTypeName() const {
        using DecayedType = std::decay_t<T>;
        
        if constexpr (std::is_enum_v<DecayedType>) {
            // For enum types, check the registry first
            std::string enum_name = EnumRegistry::getInstance().getEnumName<DecayedType>();
            return enum_name.empty() ? "unknown_enum" : enum_name;
        } else {
            // Use the type name resolution from argument_types.h
            return std::string(get_type_name<DecayedType>());
        }
    }

public:

    template <typename T>
    explicit inline Argument(const std::string name, T&& val)
        : arg_name(std::move(name)) {
        if constexpr (std::is_enum_v<std::decay_t<T>>) {
            type_name = "enum"; // Enum types are stored as integers by default
            
            enum_type_name = getTypeName<T>();
            if (enum_type_name != "unknown_enum") {
                values.push_back(static_cast<int>(val));
            } else {
                THROW("Unknown enum type for argument " + arg_name)
            }
        } else if constexpr (is_vector_type_v<std::decay_t<T>>) {
            using ElementType = typename std::decay_t<T>::value_type;
            std::string element_type_name = getTypeName<ElementType>();
            if (element_type_name != "unknown") {
                type_name = element_type_name;                
                is_vector = true;
                values.reserve(val.size()); // Pre-allocate for better performance
                for (auto&& v : std::forward<T>(val)) {
                    values.push_back(static_cast<ElementType>(std::forward<decltype(v)>(v)));
                }
            } else {
                THROW("Unknown vector element type for argument " + arg_name)
            }
        } else {
            type_name = getTypeName<T>();
            if (type_name != "unknown") {
                if constexpr (std::is_same_v<std::decay_t<T>, const char*>) {
                    values.push_back(std::string(val));
                } else {
                    values.push_back(static_cast<std::decay_t<T>>(std::forward<T>(val)));
                }
            } else {
                THROW("Unknown type " + std::string(typeid(T).name()) + " for argument " + arg_name)
            }
        }
    }

    // Used to store the feature key map
    explicit inline Argument(std::string name, std::map<std::string, std::string> val)
        : arg_name(std::move(name)) {
        type_name = "map_string";
        is_vector = true;
        if (!val.empty()) {
            values.reserve(val.size() * 2); // Pre-allocate for key-value pairs
            for (auto&& pair : std::move(val)) {
                values.push_back(std::move(pair.first));   // Push key
                values.push_back(std::move(pair.second));  // Push value
            }
        }
    }

    // Used to store the shared_ptr
    template <typename T>
    explicit inline Argument(std::string name, std::shared_ptr<T> val)
        : arg_name(std::move(name)) {
        type_name = "shared_ptr";

        // For MetadataReader case store an empty value
        // During deserialization the MetadataReader should be created and passed from the MasterGraph.
        if (arg_name == "meta_data_reader") {
            values.push_back(static_cast<int>(0));
        }
        // Could store additional shared_ptr metadata here if needed
    }

    // Deduces the type of parameter of the argument
    inline void extract_param(const RocalParameterType param_type, pParam parameter) {
        if (param_type == RocalParameterType::DETERMINISTIC) {
            enum_type_name = "SimpleParameter";
        } else if (param_type == RocalParameterType::RANDOM_UNIFORM) {
            enum_type_name = "UniformRand";
        } else if (param_type == RocalParameterType::RANDOM_CUSTOM) {
            enum_type_name = "CustomRand";
        }
        param = parameter;
        is_parameter = true;
    }

    // Constructor for FloatParam arguments
    explicit inline Argument(std::string name, FloatParam* param)
        : arg_name(std::move(name)) {
        type_name = "float";
        if (param == nullptr) {
            is_null_ptr = true;
            type_name = "nullptr";
            return;
        }
        extract_param(param->type, param);
    }

    // Constructor for IntParam arguments
    explicit inline Argument(std::string name, IntParam* param)
        : arg_name(std::move(name)) {
        type_name = "int";
        if (param == nullptr) {
            type_name = "nullptr";
            is_null_ptr = true;
            return;
        }
        extract_param(param->type, param);
    }
};
