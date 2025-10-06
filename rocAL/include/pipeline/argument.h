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
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "pipeline/argument_types.h"
#include "pipeline/commons.h"
#include "pipeline/enum_registry.h"
#include "parameters/parameter_factory.h"

/**
 * @brief Argument class stores the details of each argument in the Node
 * 
 * This class encapsulates argument information for pipeline nodes, supporting
 * various data types including basic types, enums, vectors, maps, and parameters.
 */
class Argument {
public:
    // Public member variables
    std::string arg_name;                 ///< Name of the argument
    std::string type_name;                ///< Denotes the data type of the argument
    std::string sub_type_name;            ///< Denotes the name of the enum/ type of parameter
    bool is_vector = false;               ///< True if the argument contains vector data
    bool is_parameter = false;            ///< True if the argument is a parameter object
    bool is_null_ptr = false;             ///< True if the argument represents a null pointer
    std::vector<std::any> values;         ///< Storage for argument values
    pParam param;                         ///< Parameter stored for parameter-type arguments

    // Constructors

    /**
     * @brief Unified template constructor for all data types
     * @tparam T The type of the value being stored
     * @param name The name of the argument
     * @param val The value to store
     * @throws std::runtime_error if the type is unknown or unsupported
     */
    template <typename T>
    explicit Argument(std::string name, T&& val) : arg_name(std::move(name)) {

        if constexpr (std::is_enum_v<std::decay_t<T>>) {
            constructFromEnum(std::forward<T>(val));
        } else if constexpr (is_vector_type_v<std::decay_t<T>>) {
            constructFromVector(std::forward<T>(val));
        } else if constexpr (is_shared_ptr_v<std::decay_t<T>>) {
            constructFromSharedPtr(std::forward<T>(val));
        } else if constexpr (is_string_map_v<std::decay_t<T>>) {
            constructFromMap(std::forward<T>(val));
        } else if constexpr (is_param_type_v<std::decay_t<T>>) {
            constructFromParam(std::forward<T>(val));
        } else {
            constructFromBasicType(std::forward<T>(val));
        }
    }


private:
    /**
     * @brief Helper method to get type name from registry or built-in types
     * @tparam T The type to get the name for
     * @return The string representation of the type name
     */
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

    /**
     * @brief Constructs argument from enum type
     * @tparam T The enum type
     * @param val The enum value
     */
    template<typename T>
    void constructFromEnum(T&& val) {
        static_assert(std::is_enum_v<std::decay_t<T>>, "T must be an enum type");
        
        type_name = "enum";
        sub_type_name = getTypeName<T>();
        
        if (sub_type_name != "unknown_enum") {
            values.push_back(static_cast<int>(val));
        } else {
            THROW("Unknown enum type for argument " + arg_name);
        }
    }

    /**
     * @brief Constructs argument from vector type
     * @tparam T The vector type
     * @param val The vector value
     */
    template<typename T>
    void constructFromVector(T&& val) {
        static_assert(is_vector_type_v<std::decay_t<T>>, "T must be a vector type");
        
        using ElementType = typename std::decay_t<T>::value_type;
        const std::string element_type_name = getTypeName<ElementType>();
        
        if (element_type_name != "unknown") {
            type_name = element_type_name;
            is_vector = true;
            values.reserve(val.size());
            
            auto&& local_val = std::forward<T>(val);
            for (auto&& v : local_val) {
                values.push_back(static_cast<ElementType>(std::forward<decltype(v)>(v)));
            }
        } else {
            THROW("Unknown vector element type for argument " + arg_name);
        }
    }

    /**
     * @brief Constructs argument from basic type
     * @tparam T The basic type
     * @param val The value
     */
    template<typename T>
    void constructFromBasicType(T&& val) {
        type_name = getTypeName<T>();
        
        if (type_name != "unknown") {
            if constexpr (std::is_same_v<std::decay_t<T>, const char*>) {
                values.push_back(std::string(val));
            } else {
                values.push_back(static_cast<std::decay_t<T>>(std::forward<T>(val)));
            }
        } else {
            THROW("Unknown type " + std::string(typeid(T).name()) + " for argument " + arg_name);
        }
    }

    /**
     * @brief Constructs argument from shared pointer type
     * @tparam T The shared_ptr type
     * @param val The shared_ptr value
     */
    template<typename T>
    void constructFromSharedPtr(T&& val) {
        static_assert(is_shared_ptr_v<std::decay_t<T>>, "T must be a shared_ptr type");
        
        type_name = "shared_ptr";
        // For MetaDataReader, mark as an external reference to be resolved during deserialization.
        // The actual MetaDataReader instance should be created and provided by the MasterGraph/Pipeline.
        if (arg_name == "meta_data_reader") {
            sub_type_name = "MetaDataReader";
            // No serialized payload for external references.
        } else {
            THROW("Unsupported shared_ptr type for argument " + arg_name);
        }
    }

    /**
     * @brief Constructs argument from string-to-string map type
     * @tparam T The map type
     * @param val The map value
     */
    template<typename T>
    void constructFromMap(T&& val) {
        static_assert(is_string_map_v<std::decay_t<T>>, "T must be a string-to-string map type");
        
        type_name = "map_string";
        is_vector = true;
        
        if (!val.empty()) {
            values.reserve(val.size() * 2); // Pre-allocate for key-value pairs
            auto&& string_map = std::forward<T>(val);
            for (auto&& pair : string_map) {
                values.push_back(pair.first);   // Push key
                values.push_back(pair.second);  // Push value
            }
        }
    }

    /**
     * @brief Constructs argument from parameter type (FloatParam* or IntParam*)
     * @tparam T The parameter pointer type
     * @param val The parameter pointer value
     */
    template<typename T>
    void constructFromParam(T&& val) {
        static_assert(is_param_type_v<std::decay_t<T>>, "T must be a parameter pointer type");
        
        using DecayedType = std::decay_t<T>;
        
        if constexpr (std::is_same_v<DecayedType, FloatParam*>) {
            type_name = "float";
        } else if constexpr (std::is_same_v<DecayedType, IntParam*>) {
            type_name = "int";
        }
        
        if (val == nullptr) {
            is_null_ptr = true;
            type_name = "nullptr";
            return;
        }
        
        extractParam(val->type, val);
    }

    /**
     * @brief Deduces the type of parameter of the argument
     * @param param_type The type of the parameter
     * @param parameter The parameter object
     */
    void extractParam(RocalParameterType param_type, pParam parameter) {
        switch (param_type) {
            case RocalParameterType::DETERMINISTIC:
                sub_type_name = "SimpleParameter";
                break;
            case RocalParameterType::RANDOM_UNIFORM:
                sub_type_name = "UniformRand";
                break;
            case RocalParameterType::RANDOM_CUSTOM:
                sub_type_name = "CustomRand";
                break;
            default:
                THROW("Unknown parameter type for argument " + arg_name);
        }
        param = parameter;
        is_parameter = true;
    }
};
