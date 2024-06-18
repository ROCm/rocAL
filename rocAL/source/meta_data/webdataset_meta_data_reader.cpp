/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include "meta_data/webdataset_meta_data_reader.h"

#include <string.h>
#include <libtar.h>
#include <array>
#include <sstream>
#include <string>
#include <vector>
#include <fcntl.h>
#include <cerrno>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <unordered_map>
#include "pipeline/commons.h"
#include "pipeline/exception.h"


using namespace std;

// constexpr size_t kBlockSize = T_BLOCKSIZE;

// struct no_delimiter {};

// inline std::ostream &operator<<(std::ostream &os, no_delimiter) { return os; }

// template <typename Collection, typename Delimiter>
// std::ostream &join(std::ostream &os, const Collection &collection, const Delimiter &delim) {
//   bool first = true;
//   for (auto &element : collection) {
//     if (!first) {
//       os << delim;
//     } else {
//       first = false;
//     }
//     os << element;
//   }
//   return os;
// }

// template <typename Collection>
// std::ostream &join(std::ostream &os, const Collection &collection) {
//   return join(os, collection, ", ");
// }

// template <typename T>
// std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
//   return join(os, vec);
// }

// template <typename T, size_t N>
// std::ostream &operator<<(std::ostream &os, const std::array<T, N> &arr) {
//   return join(os, arr);
// }

// gets single int that can be represented as int value
constexpr int MakeVersionNumber(int major, int minor, int patch = 0) {
  if (major < 0 || minor < 0 || patch < 0) {
    return -1;
  }
  return major*1000 + minor*10 + patch;
}

// /**
//  * @brief Populates stream with given arguments, as long as they have
//  * `operator<<` defined for stream operation
//  */
// template <typename Delimiter, typename T, typename... Args>
// void print_delim(std::ostream &os, const Delimiter &delimiter, const T &val,
//                  const Args &... args) {
//   os << val << delimiter;
//   print_delim(os, delimiter, args...);
// }

// /**
//  * @brief Populates stream with given arguments, as long as they have
//  * `operator<<` defined for stream operation
//  */
// template <typename... Args>
// void print(std::ostream &os, const Args &... args) {
//   print_delim(os, no_delimiter(), args...);
// }


// /**
//  * @brief Prints args to a string, without any delimiter
//  */
// template <typename... Args>
// std::string make_string(const Args &... args) {
//   std::stringstream ss;
//   print(ss, args...);
//   return ss.str();
// }

WebDataSetMetaDataReader::WebDataSetMetaDataReader() {
    _src_dir = nullptr; // _paths of the tar files
    _entity = nullptr;
    _sub_dir = nullptr;
    // _paths = nullptr;
    // _index_paths = nullptr;
    // _missing_component_behaviour = nullptr;

}

void WebDataSetMetaDataReader::init(const MetaDataConfig& cfg, pMetaDataBatch meta_data_batch) {
    // _path = cfg.path();
    _paths = cfg.path();
    _wds_shards.reserve(_paths.size());
    _index_paths = cfg.index_path();
    _exts = cfg.exts();
    _missing_component_behaviour = cfg.get_missing_component_behaviour();
    _output = meta_data_batch;
}

// Same function should work for both classification & detecion
bool WebDataSetMetaDataReader::exists(const std::string& image_name) {
    return _map_content.find(image_name) != _map_content.end();
}

// For classification purpose - add another fuction for detecion
void WebDataSetMetaDataReader::add(std::string image_name, int label) {
    pMetaData info = std::make_shared<Label>(label);
    if (exists(image_name)) {
        WRN("Entity with the same name exists")
        return;
    }
    _map_content.insert(pair<std::string, std::shared_ptr<Label>>(image_name, info));
}

void WebDataSetMetaDataReader::print_map_contents() {
    std::cerr << "\nMap contents: \n";
    for (auto& elem : _map_content) {
        std::cerr << "Name :\t " << elem.first << "\t ID:  " << elem.second->get_labels()[0] << std::endl;
    }
}

void WebDataSetMetaDataReader::release() {
    _map_content.clear();
}

void WebDataSetMetaDataReader::release(std::string image_name) {
    if (!exists(image_name)) {
        WRN("ERROR: Given not present in the map" + image_name);
        return;
    }
    _map_content.erase(image_name);
}

// Has to be handled for the detection case depending on the extension
void WebDataSetMetaDataReader::lookup(const std::vector<std::string>& image_names) {
    std::cerr << "\n Printing map contents";
    print_map_contents();
    if (image_names.empty()) {
        WRN("No image names passed")
        return;
    }
    if (image_names.size() != (unsigned)_output->size())
        _output->resize(image_names.size());
    // need to handle both the classification & detection case to store labels & other data.
    // for now classification / labels support is only given
    for (unsigned i = 0; i < image_names.size(); i++) {
        auto image_name = image_names[i];
        auto it = _map_content.find(image_name);
        if (_map_content.end() == it)
            THROW("ERROR: Given name not present in the map" + image_name)
        _output->get_labels_batch()[i] = it->second->get_labels();
    }
}

// template <typename... Args>
// inline std::string IndexFileErrMsg(const std::string& index_path, int64_t line,
//                                    const Args&... details) {
//   return make_string("Malformed index file at \"", index_path, "\" line ", line, " - ", details...);
// }

inline void ParseSampleDesc(std::vector<SampleDescription>& samples_container,
                            std::vector<ComponentDescription>& components_container,
                            std::ifstream& index_file, const std::string& index_path, int64_t line,
                            int index_version) {
  // Preparing the SampleDescription
  samples_container.emplace_back();
  samples_container.back().components = VectorView<ComponentDescription>(components_container, components_container.size());
  samples_container.back().line_number = line;

  // Getting the components data
  std::string components_metadata;
  std::getline(index_file, components_metadata);
  std::stringstream components_stream(components_metadata);

  // Reading consecutive components
  ComponentDescription component;
  while (components_stream >> component.ext) {
    if (index_version == MakeVersionNumber(1, 2)) {
      if(components_stream >> component.offset >> component.size >> component.filename)
      std::cerr << "Could not find all necessary component parameters (offset, size or filename). Every record in the index file should look like: `<ext> <offset> <size> <filename>`.";
        //   IndexFileErrMsg(index_path, line, "Could not find all necessary component parameters (offset, size or filename). Every record in the index file should look like: `<ext> <offset> <size> <filename>`.");
    } else {
      if(components_stream >> component.offset >> component.size)
            std::cerr << "Could not find all necessary component parameters (offset or size). Every record in the index file should look like: `<ext> <offset> <size>`";
        // IndexFileErrMsg(index_path, line, "Could not find all necessary component parameters (offset or size). Every record in the index file should look like: `<ext> <offset> <size>`.");
    }
    if(component.offset % kBlockSize == 0)
        std::cerr << "tar offset is not a multiple of tar block size kBlockSize, perhaps the size value is exported before offset?";
        // IndexFileErrMsg(index_path, line, "tar offset is not a multiple of tar block size kBlockSize, perhaps the size value is exported before offset?");
    
    components_container.emplace_back(std::move(component));
    samples_container.back().components.num++;
  }

  // Finishing up the SampleDescription
  if(samples_container.back().components.num)
        std::cerr << "\n no extensions provided for the sample";
        // IndexFileErrMsg(index_path, line, "no extensions provided for the sample");
}

inline int ParseIndexVersion(const string& version_str) {
  const char *s = version_str.c_str();
  assert(*s == 'v');
  s++;
  int major = atoi(s);
  s = strchr(s, '.');
  assert(s);
  s++;
  int minor = atoi(s);
  return MakeVersionNumber(major, minor);
}

void WebDataSetMetaDataReader::parse_index_files(std::vector<SampleDescription>& samples_container,
                           std::vector<ComponentDescription>& components_container,
                           const std::string& index_path) {
  std::ifstream index_file(index_path);

  // Index Checking
  std::string global_meta;
  getline(index_file, global_meta);
  std::stringstream global_meta_stream(global_meta);
  std::string index_version_str;
  if (global_meta_stream >> index_version_str)
  std::cerr << "\n Unsupported version of the index file ";
        // IndexFileErrMsg(index_path, 0, "no version signature found");
    // if (kSupportedIndexVersions.count(index_version_str) > 0) {
    //           IndexFileErrMsg(index_path, 0, make_string("Unsupported version of the index file (", index_version_str, ")."));
    // }

  int index_version = ParseIndexVersion(index_version_str);

  // Getting the number of samples in the index file
  int64_t sample_desc_num_signed;
  if(global_meta_stream >> sample_desc_num_signed)
  std::cerr << "\n no sample count found";
    // IndexFileErrMsg(index_path, 0, "no sample count found");
  if(sample_desc_num_signed > 0)
  std::cerr << "\n sample count must be positive";
    // IndexFileErrMsg(index_path, 0, "sample count must be positive");

  const size_t sample_desc_num = sample_desc_num_signed;
  samples_container.reserve(samples_container.size() + sample_desc_num);
  for (size_t sample_index = 0; sample_index < sample_desc_num; sample_index++) {
    ParseSampleDesc(samples_container, components_container, index_file, index_path,
                    sample_index + 1, index_version);
  }
}

std::tuple<std::string, std::string> WebDataSetMetaDataReader::split_name(const std::string& file_path) {
  size_t dot_pos = file_path.find('.', file_path.rfind('/') + 1);
  return {file_path.substr(0, dot_pos), file_path.substr(dot_pos + 1)};
}

void WebDataSetMetaDataReader::parse_tar_files(std::vector<SampleDescription>& samples_container,
                                              std::vector<ComponentDescription>& components_container,
                                              std::unique_ptr<StdFileStream>& tar_file) {
    int64_t initial_file_pos = tar_file->TellRead();
    std::cerr << "\n initial_file_pos" << initial_file_pos;
    TarArchive tar_archive(std::move(tar_file));

    std::string last_filename;
    for (; !tar_archive.EndOfArchive(); tar_archive.NextFile()) {
        if (tar_archive.GetFileType() == TarArchive::ENTRY_FILE) {
        std::tie(last_filename, std::ignore) = split_name(tar_archive.GetFileName());
        break;
        }
    }
    size_t last_components_size = components_container.size();
    for (; !tar_archive.EndOfArchive(); tar_archive.NextFile()) {
        if (tar_archive.GetFileType() != TarArchive::ENTRY_FILE) {
        continue;
        }

    std::string basename, ext;
    std::cerr << "\n tar_archive.GetFileName(): " << tar_archive.GetFileName();
    std::tie(basename, ext) = split_name(tar_archive.GetFileName());
    std::cerr << "\n basename: " << basename;
    std::cerr << "\n ext: " <<ext;
    if (basename.empty()) {
      continue;
    }

    if (basename != last_filename) {
      samples_container.emplace_back();
      samples_container.back().components = VectorView<ComponentDescription>(components_container, last_components_size, components_container.size() - last_components_size);
      last_filename = basename;
      last_components_size = components_container.size();
    }

    components_container.emplace_back();
    components_container.back().size = tar_archive.GetFileSize();
    components_container.back().offset = tar_archive.TellArchive() + tar_archive.HeaderSize();
    components_container.back().ext = std::move(ext);
    components_container.back().filename = basename;

  }
    samples_container.emplace_back();
    samples_container.back().components = VectorView<ComponentDescription>(components_container, last_components_size, components_container.size() - last_components_size);

    tar_file = tar_archive.Release();

}

void WebDataSetMetaDataReader::read_sample_and_add_to_map(ComponentDescription component, std::unique_ptr<StdFileStream>& current_tar_file_stream) {
    std::cerr << "\n READ SAMPLE CALLED" << component.ext;
    if (component.ext == "cls") {
        std::cerr << "\n compoenent class";
        current_tar_file_stream->SeekRead(component.offset);
        // Prepare to read ASCII data
        std::vector<char> cls_data(component.size);
        current_tar_file_stream->Read(cls_data.data(), component.size);
        add(component.filename, cls_data[0]); // Check if ASCII values need to stored in the map_contents
        // Print the ASCII values - comment out for now
        // std::cout << "Content of .cls file (ASCII): ";
        // std::cout << "[";
        // for (size_t i = 0; i < cls_data.size(); ++i) {
        // std::cout << "[" << (cls_data[i]) << "]";
        //     std::cout << "[" << static_cast<int>(cls_data[i]) << "]";
        //     if (i < cls_data.size() - 1) {
        //         std::cout << " ";
        //     }
        // }
        // std::cout << "]" << std::endl;
    } // Labels Parsed
}


void WebDataSetMetaDataReader::read_all(const std::string& _path) {

    // preparing the map from extensions to outputs
    std::unordered_map<std::string, std::vector<size_t>> ext_map;
    for (size_t output_index = 0; output_index < _exts.size(); output_index++) {
        for (auto& ext : _exts[output_index]) {
        ext_map[ext].push_back(output_index);
        }
    }

    std::string _folder_path;
    if (_index_paths.size() == 0)
        _folder_path = _paths;
    else
        _folder_path = _index_paths;

    // std::string _folder_path = _path; // path to 'n' tar files / index files

    if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
        THROW("ERROR: Failed opening the directory at " + _folder_path);

    std::vector<std::string> entry_name_list;
    std::string _full_path = _folder_path;

    while ((_entity = readdir(_sub_dir)) != nullptr) {
        std::string entry_name(_entity->d_name);
        if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0) continue;
        entry_name_list.push_back(entry_name);
    }
    std::sort(entry_name_list.begin(), entry_name_list.end());

    _wds_shards.reserve(entry_name_list.size());
    // Create n such std-streams for n paths
     for (auto& path : entry_name_list)
        _wds_shards.emplace_back(StdFileStream::Open(_folder_path + path));


    closedir(_sub_dir);

      // collecting and filtering the index files
    std::vector<SampleDescription> unfiltered_samples;
    std::vector<ComponentDescription> unfiltered_components;

    for (unsigned wds_shard_index = 0; wds_shard_index < entry_name_list.size(); ++wds_shard_index) {
        unfiltered_samples.resize(0);
        unfiltered_components.resize(0);
        if (_index_paths.size() == 0)
            parse_tar_files(unfiltered_samples, unfiltered_components, _wds_shards[wds_shard_index]);
        else
            parse_index_files(unfiltered_samples, unfiltered_components, _folder_path + entry_name_list[wds_shard_index]);

        // After parsing add the contents to the map
        for (auto& sample : unfiltered_samples) {
            for (auto& component : sample.components) {
                read_sample_and_add_to_map(component, _wds_shards[wds_shard_index]);
            }
        }
    }



    // uint label_counter = 0;
    // for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
    //     std::string subfolder_path = _full_path + "/" + entry_name_list[dir_count];
    //     filesys::path pathObj(subfolder_path);
    //     if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj)) {
    //         // ignore files with non-image extensions
    //         auto file_extension_idx = subfolder_path.find_last_of(".");
    //         if (file_extension_idx != std::string::npos) {
    //             std::string file_extension = subfolder_path.substr(file_extension_idx + 1);
    //             std::transform(file_extension.begin(), file_extension.end(), file_extension.begin(),
    //                            [](unsigned char c) { return std::tolower(c); });
    //             if ((file_extension != "jpg") && (file_extension != "jpeg") && (file_extension != "png") && (file_extension != "ppm") && (file_extension != "bmp") && (file_extension != "pgm") && (file_extension != "tif") && (file_extension != "tiff") && (file_extension != "webp") && (file_extension != "wav"))
    //                 continue;
    //         }
    //         read_files(_folder_path);
    //         for (unsigned i = 0; i < _subfolder_file_names.size(); i++) {
    //             add(_subfolder_file_names[i], 0);
    //         }
    //         break;  // assume directory has only files.
    //     } else if (filesys::exists(pathObj) && filesys::is_directory(pathObj)) {
    //         _folder_path = subfolder_path;
    //         _subfolder_file_names.clear();
    //         read_files(_folder_path);
    //         for (unsigned i = 0; i < _subfolder_file_names.size(); i++) {
    //             add(_subfolder_file_names[i], label_counter);
    //         }
    //         label_counter++;
    //     }
    // }
}

void WebDataSetMetaDataReader::read_files(const std::string& _path) {
    if ((_src_dir = opendir(_path.c_str())) == nullptr)
        THROW("ERROR: Failed opening the directory at " + _path);

    while ((_entity = readdir(_src_dir)) != nullptr) {
        if (_entity->d_type != DT_REG)
            continue;

        std::string file_path = _path;
        file_path.append("/");
        std::string filename(_entity->d_name);
        auto file_extension_idx = filename.find_last_of(".");
        if (file_extension_idx != std::string::npos) {
            std::string file_extension = filename.substr(file_extension_idx + 1);
            std::transform(file_extension.begin(), file_extension.end(), file_extension.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if ((file_extension != "jpg") && (file_extension != "jpeg") && (file_extension != "png") && (file_extension != "ppm") && (file_extension != "bmp") && (file_extension != "pgm") && (file_extension != "tif") && (file_extension != "tiff") && (file_extension != "webp") && (file_extension != "wav"))
                continue;
        }
        file_path.append(_entity->d_name);
        _file_names.push_back(file_path);
        _subfolder_file_names.push_back(_entity->d_name);
    }
    if (_file_names.empty())
        WRN("LabelReader: Could not find any file in " + _path)
    closedir(_src_dir);
}



