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

    /**
     * @brief Retrieves the value of the argument as the specified type.
     * 
     * This template method extracts the stored value(s) from the argument and returns
     * them as the requested type. It supports scalar types, vectors, maps, and parameter
     * pointer types.
     * 
     * @tparam T The type to which the argument value should be cast.
     * @return The value of the argument as type T.
     * @note For parameter pointer types (FloatParam*, IntParam*), nullptr is returned
     *       if the argument is a null pointer.
     */
    template <typename T>
    T get() const {
        // Handle parameter pointer types (FloatParam* or IntParam*)
        if constexpr (std::is_same_v<T, FloatParam*> || std::is_same_v<T, IntParam*>) {
            // Return nullptr if the parameter was null
            if (is_null_ptr) {
                return nullptr;
            }
            if constexpr (std::is_same_v<T, FloatParam*>)
                return std::get<FloatParam*>(param);
            else if constexpr (std::is_same_v<T, IntParam*>)
                return std::get<IntParam*>(param);
        } 
        // Handle string-to-string map type
        else if constexpr (std::is_same_v<T, std::map<std::string, std::string>>) {
            // Validate that we have key-value pairs (even number of elements)
            if ((values.size() % 2) != 0)
                THROW("Corrupted map payload for argument : " + arg_name + ".");
            // Reconstruct the map from alternating key-value elements in the values vector
            std::map<std::string, std::string> feature_map;
            for (size_t i = 0; i < values.size(); i += 2) {
                const auto& key = std::any_cast<const std::string&>(values[i]);
                const auto& value = std::any_cast<const std::string&>(values[i + 1]);
                feature_map.emplace(key, value);
            }
            return feature_map;
        } 
        // Handle all other types (scalars, vectors, etc.)
        else {
            if (is_null_ptr || is_parameter)
                THROW("Type mismatch: cannot retrieve non-parameter type from a parameter argument (arg_name: '" + arg_name + "', type_name: '" + type_name + "')");

            // Handle vector types - reconstruct vector from stored elements
            if constexpr (is_vector_type<std::decay_t<T>>::value) {
                using ElementType = typename std::decay_t<T>::value_type;

                std::vector<ElementType> result;
                result.reserve(values.size());
                for (const auto& v : values) {
                    result.push_back(std::any_cast<ElementType>(v));
                }
                return result;
            } 
            // Handle scalar types - return the single stored value
            else if (!is_vector) {
                if (values.empty()) {
                    THROW("Value not present for the given argument : " + arg_name + ".");
                }
                return std::any_cast<T>(values[0]);
            } 
            else {
                THROW("Unsupported type requested for argument : " + arg_name + " of type " + type_name);
            }
        }
    }

    // Constructors
    /**
     * @brief Default constructor for Argument.
     */
    Argument() {}

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

/// @brief Class to manage a set of Arguments for a Node
class ArgumentSet {

public:
    ArgumentSet() = default;
    
    // Iterator support for range-based for loops
    using iterator = std::unordered_map<std::string, Argument>::iterator;
    using const_iterator = std::unordered_map<std::string, Argument>::const_iterator;
    
    iterator begin() { return args_map.begin(); }
    iterator end() { return args_map.end(); }
    const_iterator begin() const { return args_map.begin(); }
    const_iterator end() const { return args_map.end(); }

    void add_argument(const std::string& name, const Argument& arg) {
        args_map[name] = arg;
    }

    template<typename T>
    void add_new_argument(const std::string& name, T&& value) {
        args_map[name] = Argument(name, std::forward<T>(value));
    }
    
    void clear() {
        args_map.clear();
    }

    const Argument& get_argument(const std::string& name) const {
        auto it = args_map.find(name);
        if (it != args_map.end()) {
            return it->second;
        } else {
            THROW("Argument " + name + " not found");
        }
    }

    template<typename T>
    T get(const std::string& name) const {
        const Argument& arg = get_argument(name);
        try {  
            return arg.get<T>();
        } catch (const std::bad_any_cast& e) {
            THROW("Type mismatch when retrieving value for argument " + name + ": " + e.what());
        }
    }

    size_t size() const {
        return args_map.size();
    }

private:
    std::unordered_map<std::string, Argument> args_map;
};

template <typename... Args, std::size_t... I>
std::tuple<Args...> unpack_arguments_impl(const ArgumentSet& arguments, const std::vector<std::string> &arg_names, 
                                          std::index_sequence<I...>) {
    return std::make_tuple(arguments.get<Args>(arg_names[I])...);
}

/**
 * @brief Extracts arguments into a tuple using index sequence expansion
 * 
 * This helper function unpacks a vector of Argument objects into a std::tuple
 * with the specified types. It uses compile-time index sequences to extract
 * each argument at the corresponding position and cast it to the requested type.
 */
template <typename... Args>
std::tuple<Args...> unpack_arguments(const ArgumentSet& arguments, const std::vector<std::string> &arg_names) {
    return unpack_arguments_impl<Args...>(arguments, arg_names, std::index_sequence_for<Args...>{});
}

/**
 * @brief Initializes a node by unpacking and applying arguments from a vector of Argument objects
 * 
 * This template function provides type-safe argument deserialization for node initialization.
 * It extracts typed values from the Argument vector, validates the argument count, and applies
 * them to the node's init() method.
 * 
 * @tparam NodeType The type of node to initialize (e.g., BrightnessNode, ImageLoaderNode)
 * @tparam Args Variadic template parameters representing the expected argument types
 * @param node Pointer to the node instance to initialize
 * @param arguments Vector of Argument objects containing the serialized argument values
 * @return true if initialization succeeded, false if argument count mismatch or type conversion failed
 * 
 * @example
 * // For a node expecting (float, float) arguments:
 * if (init_args<BrightnessNode, float, float>(this, arguments)) return;
 * // For a node expecting (FloatParam*, FloatParam*) arguments:
 * if (init_args<BrightnessNode, FloatParam*, FloatParam*>(this, arguments)) return;
 */
template <typename NodeType, typename... Args>
bool init_args(NodeType* node, const std::vector<std::string> &arg_names, const ArgumentSet& arguments) {

    if (arguments.size() != sizeof...(Args)) {
        THROW("Argument count mismatch: expected " + std::to_string(sizeof...(Args)) + 
              " but got " + std::to_string(arguments.size()));
    }
    try {
        // Unpack arguments with type-check and casting
        auto unpacked_args = unpack_arguments<Args...>(arguments, arg_names);

        std::apply([&](Args&... unpacked) {
            node->init(unpacked...);
        }, unpacked_args);

        return true;
    } catch (const std::exception& e) {
        return false; // Type mismatch
    }
}
