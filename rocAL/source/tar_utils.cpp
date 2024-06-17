#include "tar_utils.h"
#include <mutex>
#include <vector>
#include <list>
#include <cassert>
// #include <cmath>
#include <cstdarg>
#include <algorithm>
#include <cstring>
#include "pipeline/exception.h"

template <typename value, typename alignment>
constexpr value align_up(value v, alignment a) {
  return v + ((a - 1) & -v);
}

constexpr uint64_t operator ""_u64(unsigned long long x) {  // NOLINT(runtime/int)
    return x;
}

constexpr uint64_t kEmptyEofBlocks = 2;
constexpr uint64_t kTarArchiveBufferInitSize = 1;

std::mutex instances_mutex;
std::list<std::vector<TarArchive*>> instances_registry = {
    std::vector<TarArchive*>(kTarArchiveBufferInitSize)
  };
TarArchive** instances = instances_registry.back().data();

int Register(TarArchive* archive) {
  std::lock_guard<std::mutex> instances_lock(instances_mutex);
  for (auto& instances_entry : instances_registry.back()) {
    if (instances_entry != nullptr) {
      continue;
    }
    instances_entry = archive;
    return &instances_entry - instances;
  }
  std::vector<TarArchive*>& old = instances_registry.back();
  instances_registry.emplace_back();
  std::vector<TarArchive*>& curr = instances_registry.back();
  curr.reserve(old.size() * 2);
  curr = old;
  curr.push_back(archive);
  curr.resize(old.size() * 2, nullptr);
  instances = curr.data();
  return old.size();
}

inline void Unregister(int _instance_handle) {
  instances[_instance_handle] = nullptr;
}

inline TAR* ToTarHandle(void* handle) {
  return reinterpret_cast<TAR*>(handle);
}

inline TAR** ToTarHandle(void** handle) {
  return reinterpret_cast<TAR**>(handle);
}

ssize_t LibtarReadTarArchive(int _instance_handle, void* buf, size_t count) {
  const auto current_archive = instances[_instance_handle];
  const ssize_t num_read = current_archive->_stream->Read(reinterpret_cast<uint8_t*>(buf), count);
  return num_read;
}

int LibtarOpenTarArchive(const char*, int oflags, ...) {
  va_list args;
  va_start(args, oflags);
  const int _instance_handle = va_arg(args, int);
  va_end(args);
  return _instance_handle;
}

static tartype_t kTarArchiveType = {LibtarOpenTarArchive, [](int) -> int { return 0; },
                                    LibtarReadTarArchive,
                                    [](int, const void*, size_t) -> ssize_t { return 0; }};

TarArchive::TarArchive(std::unique_ptr<StdFileStream> stream)
    : _stream(std::move(stream)), _instance_handle(Register(this)) {
  tar_open(ToTarHandle(&_handle), "", &kTarArchiveType, 0, _instance_handle, TAR_GNU);
  _stream->SeekRead(0);
  _eof = _stream->Size() == 0;
  ParseHeader();
}

TarArchive::TarArchive(TarArchive&& other) {
  *this = std::move(other);
}

TarArchive::~TarArchive() {
  Release();
}

TarArchive& TarArchive::operator=(TarArchive&& other) {
  if (&other != this) {
    _stream = std::move(other._stream);
    std::swap(_handle, other._handle);
    std::swap(_filename, other._filename);
    std::swap(_filesize, other._filesize);
    std::swap(_filetype, other._filetype);
    std::swap(_readoffset, other._readoffset);
    std::swap(_current_header, other._current_header);
    std::swap(_eof, other._eof);
    std::swap(_instance_handle, other._instance_handle);
    if (_instance_handle >= 0) {
      std::lock_guard<std::mutex> instances_lock(instances_mutex);
      instances[_instance_handle] = this;
    }
    other.Release();
  }
  return *this;
}

constexpr size_t RoundToBlockSize(size_t count) {
  return align_up(count, kBlockSize);
}

bool TarArchive::NextFile() {
  if (_eof) {
    return false;
  }

  const int64_t offset = _stream->TellRead() + RoundToBlockSize(_filesize) - _readoffset;
  _current_header = offset;
  _stream->SeekRead(offset);
  ParseHeader();
  return !_eof;
}

bool TarArchive::EndOfArchive() const {
  return _eof;
}

void TarArchive::SeekArchive(int64_t offset) {
  if (offset == _current_header) {
    return;
  }
  assert(offset % T_BLOCKSIZE == 0);
  _eof = false;
  _readoffset = 0;
  _stream->SeekRead(offset);
  _current_header = offset;
  ParseHeader();
}

int64_t TarArchive::TellArchive() const {
  return _current_header;
}

int64_t TarArchive::HeaderSize() const {
  return _stream->TellRead() - _readoffset - _current_header;
}

const std::string& TarArchive::GetFileName() const {
  return _filename;
}

size_t TarArchive::GetFileSize() const {
  return _filesize;
}

TarArchive::EntryType TarArchive::GetFileType() const {
  return _filetype;
}

std::shared_ptr<void> TarArchive::ReadFile() {
  _stream->SeekRead(_stream->TellRead() - _readoffset);
  std::shared_ptr<void> out;
  if (out != nullptr) {
    _readoffset = _filesize;
  }
  return out;
}

size_t TarArchive::Read(void *buffer, size_t count) {
  if (_eof) {
    return 0;
  }
  count = std::clamp(_filesize - _readoffset, 0_u64, count);
  size_t num_read_bytes = _stream->Read(buffer, count);
  _readoffset += num_read_bytes;
  return num_read_bytes;
}

bool TarArchive::EndOfFile() const {
  return _readoffset >= _filesize;
}

inline void TarArchive::SetEof() {
  _eof = true;
  _filename = "";
  _filesize = 0;
  _filetype = ENTRY_NONE;
}

inline void TarArchive::ParseHeader() {
  if (_eof) {
    return;
  }
  int errorcode = th_read(ToTarHandle(_handle));
  if (errorcode) {
    if (errorcode == -1)
    THROW("Corrupted tar file at " + ToTarHandle(_handle)->pathname);
    SetEof();
  } else {
    _filename = th_get_pathname(ToTarHandle(_handle));
    _filesize = th_get_size(ToTarHandle(_handle));

    if (TH_ISREG(ToTarHandle(_handle))) {
      _filetype = ENTRY_FILE;
    } else if (TH_ISDIR(ToTarHandle(_handle))) {
      _filetype = ENTRY_DIR;
    } else if (TH_ISLNK(ToTarHandle(_handle))) {
      _filetype = ENTRY_HARDLINK;
    } else if (TH_ISSYM(ToTarHandle(_handle))) {
      _filetype = ENTRY_SYMLINK;
    } else if (TH_ISCHR(ToTarHandle(_handle))) {
      _filetype = ENTRY_CHARDEV;
    } else if (TH_ISBLK(ToTarHandle(_handle))) {
      _filetype = ENTRY_BLOCKDEV;
    } else if (TH_ISFIFO(ToTarHandle(_handle))) {
      _filetype = ENTRY_FIFO;
    } else {
      /*
       * POSIX.1-2001 tar format adds additional entries containing attributes.
       * As we are interested only in parsing actual data, we mark them as unknown
       * entries, and skip when reading data.
       */
       _filetype = ENTRY_UNKNOWN;
    }
  }
  _readoffset = 0;
}

std::unique_ptr<StdFileStream> TarArchive::Release() {
  auto out = std::move(_stream);
  if (_handle != nullptr) {
    tar_close(ToTarHandle(_handle));
    _handle = nullptr;
  }
  _readoffset = 0;
  _current_header = 0;
  SetEof();
  if (_instance_handle >= 0) {
    Unregister(_instance_handle);
  }
  _instance_handle = -1;
  return out;
}