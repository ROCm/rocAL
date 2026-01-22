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
#include <type_traits>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <stdexcept>
#include "parameters/parameter_factory.h"

// Enhanced type traits, to check for the type of argument used for argument processing (C++17 compatible)

// Vector type detection
template <typename T>
struct is_vector_type : std::false_type {};

template <typename T, typename Alloc>
struct is_vector_type<std::vector<T, Alloc>> : std::true_type {};

template <typename T>
constexpr bool is_vector_type_v = is_vector_type<T>::value;

// Map type detection
template <typename T>
struct is_map_type : std::false_type {};

template <typename K, typename V, typename Compare, typename Alloc>
struct is_map_type<std::map<K, V, Compare, Alloc>> : std::true_type {};

template <typename T>
constexpr bool is_map_type_v = is_map_type<T>::value;

// Shared pointer detection
template <typename T>
struct is_shared_ptr : std::false_type {};

template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

template <typename T>
constexpr bool is_shared_ptr_v = is_shared_ptr<T>::value;

// String type detection (including const char*)
template <typename T>
struct is_string_type : std::false_type {};

template <>
struct is_string_type<std::string> : std::true_type {};

template <>
struct is_string_type<const char*> : std::true_type {};

template <>
struct is_string_type<char*> : std::true_type {};

template <typename T>
constexpr bool is_string_type_v = is_string_type<std::decay_t<T>>::value;

// String-to-string map detection
template <typename T>
struct is_string_map : std::false_type {};

template <typename Compare, typename Alloc>
struct is_string_map<std::map<std::string, std::string, Compare, Alloc>> : std::true_type {};

template <typename T>
constexpr bool is_string_map_v = is_string_map<std::decay_t<T>>::value;

// Parameter type detection
template <typename T>
struct is_param_type : std::false_type {};

template <>
struct is_param_type<FloatParam*> : std::true_type {};

template <>
struct is_param_type<IntParam*> : std::true_type {};

template <typename T>
constexpr bool is_param_type_v = is_param_type<std::decay_t<T>>::value;

// SFINAE-based type checking (C++17 compatible)
template <typename T>
using enable_if_basic_type = std::enable_if_t<
    std::is_arithmetic_v<std::decay_t<T>> || is_string_type_v<T>
>;

template <typename T>
using enable_if_enum_type = std::enable_if_t<std::is_enum_v<std::decay_t<T>>>;

template <typename T>
using enable_if_vector_type = std::enable_if_t<is_vector_type_v<std::decay_t<T>>>;

template <typename T>
using enable_if_map_type = std::enable_if_t<is_map_type_v<std::decay_t<T>>>;

template <typename T>
using enable_if_shared_ptr_type = std::enable_if_t<is_shared_ptr_v<std::decay_t<T>>>;

// Compile-time type name resolution
template <typename T>
constexpr const char* get_type_name() noexcept {
    using DecayedType = std::decay_t<T>;
    
    if constexpr (std::is_same_v<DecayedType, int>) return "int";
    else if constexpr (std::is_same_v<DecayedType, unsigned>) return "unsigned";
    else if constexpr (std::is_same_v<DecayedType, size_t>) return "size_t";
    else if constexpr (std::is_same_v<DecayedType, float>) return "float";
    else if constexpr (std::is_same_v<DecayedType, double>) return "double";
    else if constexpr (std::is_same_v<DecayedType, bool>) return "bool";
    else if constexpr (std::is_same_v<DecayedType, std::string>) return "string";
    else if constexpr (std::is_same_v<DecayedType, char*> || std::is_same_v<DecayedType, const char*>) return "char_str";
    else return "unknown";
}
