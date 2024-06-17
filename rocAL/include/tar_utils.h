
#include <libtar.h>
#include "file_stream.h"

constexpr size_t kBlockSize = T_BLOCKSIZE;

class TarArchive {
 public:
  TarArchive() = default;
  explicit TarArchive(std::unique_ptr<StdFileStream> _stream);
  TarArchive(TarArchive&&);
  ~TarArchive();
  TarArchive& operator=(TarArchive&&);
  bool NextFile();
  bool EndOfArchive() const;
  void SeekArchive(int64_t offset);
  int64_t TellArchive() const;
  int64_t HeaderSize() const;

  enum EntryType {
    ENTRY_NONE = 0,
    ENTRY_FILE,
    ENTRY_DIR,
    ENTRY_HARDLINK,
    ENTRY_SYMLINK,
    ENTRY_CHARDEV,
    ENTRY_BLOCKDEV,
    ENTRY_FIFO,
    ENTRY_UNKNOWN
  };

  const std::string& GetFileName() const;
  size_t GetFileSize() const;
  EntryType GetFileType() const;
  std::shared_ptr<void> ReadFile();
  size_t Read(void *buffer, size_t count);
  bool EndOfFile() const;
  std::unique_ptr<StdFileStream> Release();

 private:
  std::unique_ptr<StdFileStream> _stream;
  int _instance_handle = -1;
  void* _handle = nullptr;  // handle to the TAR struct
  friend ssize_t LibtarReadTarArchive(int, void*, size_t);
  std::string _filename;
  size_t _filesize = 0;
  EntryType _filetype = ENTRY_NONE;
  size_t _readoffset = 0;
  int64_t _current_header = 0;
  bool _eof = true;  // when this is true the value of _readoffset and _stream offset is undefined
  void SetEof();
  void ParseHeader();
};