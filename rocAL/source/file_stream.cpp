
#include <errno.h>
#include <sys/stat.h>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include "pipeline/exception.h"
#include "file_stream.h"
#include "pipeline/commons.h"

std::unique_ptr<StdFileStream> StdFileStream::Open(const std::string& path) {
    return std::make_unique<StdFileStream>(path);
// return std::unique_ptr<new StdFileStream(path);
}

StdFileStream::StdFileStream(const std::string& path) : _path(path) {
  fp_ = std::fopen(path.c_str(), "rb");
  std::cerr << "\n FIle POINTER SET";
  if(fp_ == nullptr)
    THROW("Could not open file " + path + ": " + std::strerror(errno));
}

StdFileStream::~StdFileStream() {
  Close();
}

void StdFileStream::Close() {
  if (fp_ != nullptr) {
    std::fclose(fp_);
    fp_ = nullptr;
  }
}

void StdFileStream::SeekRead(ptrdiff_t pos, int whence) {
  std::cerr << "\n pos" << pos;
  std::cerr << "\n whence :" << whence;
  if (fp_ == nullptr) {
        THROW("File pointer is null.");
    }
  if(std::fseek(fp_, pos, whence)!=0)
    THROW("Seek operation failed" + std::strerror(errno));
  std::cerr << "\n fseek done";
}

ptrdiff_t StdFileStream::TellRead() const {
  return std::ftell(fp_);
}

size_t StdFileStream::Read(void *buffer, size_t n_bytes) {
  size_t n_read = std::fread(buffer, 1, n_bytes, fp_);
  return n_read;
}

size_t StdFileStream::Size() const {
  struct stat sb;
  if (stat(_path.c_str(), &sb) == -1) {
    THROW("Unable to stat file " + _path + ": " + std::strerror(errno));
  }
  return sb.st_size;
}