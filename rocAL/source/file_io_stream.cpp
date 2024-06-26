
#include <errno.h>
#include <sys/stat.h>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include "pipeline/exception.h"
#include "file_io_stream.h"
#include "pipeline/commons.h"
#include <iostream>


std::unique_ptr<FileIOStream> FileIOStream::open(const std::string& path) {
    return std::make_unique<FileIOStream>(path);
}

FileIOStream::FileIOStream(const std::string& path) : _path(path) {
  _file_pointer = std::fopen(path.c_str(), "rb");
  if(_file_pointer == nullptr)
    THROW("Could not open file " + path + ": " + std::strerror(errno));
}

FileIOStream::~FileIOStream() {
  close_stream();
}

void FileIOStream::close_stream() {
  if (_file_pointer != nullptr) {
    std::fclose(_file_pointer);
    _file_pointer = nullptr;
  }
}

void FileIOStream::set_read_position(ptrdiff_t pos, int whence) {
  if (_file_pointer == nullptr) {
        THROW("File pointer is null.");
    }
  if(std::fseek(_file_pointer, pos, whence)!=0)
    THROW("Seek operation failed" + std::strerror(errno));
}

ptrdiff_t FileIOStream::get_current_read_position() const {
  return std::ftell(_file_pointer);
}

size_t FileIOStream::read_into_buffer(void *buffer, size_t n_bytes) {
  size_t read_n_bytes = std::fread(buffer, 1, n_bytes, _file_pointer);
  return read_n_bytes;
}

size_t FileIOStream::get_size() const {
  struct stat sb;
  if (stat(_path.c_str(), &sb) == -1) {
    THROW("Unable to stat file " + _path);
  }
  return sb.st_size;
}



// Define operator<< for std::unique_ptr<FileIOStream>
inline std::ostream& operator<<(std::ostream& os, const std::unique_ptr<FileIOStream>& ptr) {
    std::cout << ptr << std::endl;
}