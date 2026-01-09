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

#include <unordered_map>
#include <typeindex>
#include <string>
#include <any>
#include <functional>
#include "pipeline/exception.h"
/*!
 * \brief Centralized enum registry for automatic enum type name management
 * 
 * This singleton class provides a registry for enum types,
 * allowing automatic registration and lookup of enum type names.
 */
class EnumRegistry {
public:
    /*!
     * \brief Get the singleton instance of the enum registry
     * \return Reference to the singleton EnumRegistry instance
     */
    static EnumRegistry& getInstance() noexcept {
        static EnumRegistry instance;
        return instance;
    }

    /*!
     * \brief Register an enum type with its string name and conversion function
     * \tparam T The enum type to register (must be an enum)
     * \param name The string name to associate with the enum type
     */
    template<typename T>
    void registerEnum(const std::string& name) {
        static_assert(std::is_enum<T>::value, "T must be an enum type");
        auto type_idx = std::type_index(typeid(T));
        
        // Map 1: Type index to string name
        _enum_map[type_idx] = name;
        
        // Map 2: String name to conversion function that returns std::any
        _name_to_converter_map[name] = [](int value) -> std::any {
            return static_cast<T>(value);
        };
    }

    /*!
     * \brief Get the registered name for an enum type
     * \tparam T The enum type to look up
     * \return The registered string name for the enum type, or empty string if not found
     */
    template<typename T>
    std::string getEnumName() const {
        static_assert(std::is_enum<T>::value, "T must be an enum type");
        auto it = _enum_map.find(std::type_index(typeid(T)));
        return (it != _enum_map.end()) ? it->second : "";
    }

    /*!
     * \brief Get the registered name for a type_index
     * \param type The type_index to look up
     * \return The registered string name, or empty string if not found
     */
    std::string getEnumName(const std::type_index& type) const {
        auto it = _enum_map.find(type);
        return (it != _enum_map.end()) ? it->second : "";
    }

    /*!
     * \brief Convert integer value to enum and return as std::any
     * \param enumTypeName The name of the enum type
     * \param value The integer value to convert (must be a valid value for the target enum)
     * \return The converted enum value wrapped in std::any
     * \throws std::runtime_error if enum type is not registered
     */
    std::any convertIntToEnum(const std::string& enumTypeName, int value) const {
        auto it = _name_to_converter_map.find(enumTypeName);
        if (it != _name_to_converter_map.end()) {
            return it->second(value);  // Call the conversion function and return std::any
        } else {
            THROW("No conversion function registered for enum: " + enumTypeName);
        }
    }

    /*!
     * \brief Check if an enum type is registered
     * \param type The type_index to check
     * \return true if the enum type is registered, false otherwise
     */
    bool isEnumRegistered(const std::type_index& type) const noexcept {
        return _enum_map.find(type) != _enum_map.end();
    }

    /*!
     * \brief Check if an enum type is registered by name
     * \param enumTypeName The name of the enum type
     * \return true if the enum type is registered, false otherwise
     */
    bool isEnumRegistered(const std::string& enumTypeName) const noexcept {
        return _name_to_converter_map.find(enumTypeName) != _name_to_converter_map.end();
    }

private:
    EnumRegistry() = default;
    ~EnumRegistry() = default;
    EnumRegistry(const EnumRegistry&) = delete;
    EnumRegistry& operator=(const EnumRegistry&) = delete;

    // Map 1: Type index to string name mapping
    std::unordered_map<std::type_index, std::string> _enum_map;
    
    // Map 2: String name to conversion function mapping (returns std::any)
    std::unordered_map<std::string, std::function<std::any(int)>> _name_to_converter_map;
};

/*!
 * \brief Macro for automatic enum registration
 * \param EnumType The enum type to register
 * 
 * Uses a static variable with lambda function to ensure proper initialization timing.
 * Usage: REGISTER_ENUM(MyEnumType)
 */
#define REGISTER_ENUM(EnumType) \
    [[maybe_unused]] static bool enum_registered_##EnumType = []() { \
        EnumRegistry::getInstance().registerEnum<EnumType>(#EnumType); \
        return true; \
    }();
