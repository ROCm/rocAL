
#include "file_stream.h"
#include <libtar.h>

constexpr size_t kBlockSize = T_BLOCKSIZE;

class TarArchive {
  public:
    TarArchive() = default;
    explicit TarArchive(std::unique_ptr<FileIOStream> stream);
    TarArchive(TarArchive &&);
    ~TarArchive();
    TarArchive &operator=(TarArchive &&);
    bool advance_to_next_file_in_tar();
    bool at_end_of_archive() const;
    void seek_to_offset_in_archive(int64_t offset);
    int64_t get_current_archive_offset() const;
    int64_t get_current_header_size() const;

    enum EntryType {
        ENTRY_NONE = 0,
        ENTRY_FILE,
        ENTRY_DIR,
        ENTRY_HARDLINK,
        ENTRY_SYMLINK,
        ENTRY_CHARDEV,
        ENTRY_BLOCKDEV,
        ENTRY_FIFO,
        ENTRY_NOT_DEFINED
    };

    const std::string &get_current_file_name() const;
    size_t get_current_file_size() const;
    EntryType get_current_file_type() const;
    std::shared_ptr<void> read_current_file();
    size_t read_into_buffer(void *buffer, size_t count);
    bool is_end_of_file() const;
    std::unique_ptr<FileIOStream> release_file_stream();

  private:
    std::unique_ptr<FileIOStream> _stream;
    int _instance_handle = -1;
    void *_handle = nullptr;
    friend ssize_t read_tar_archive(int, void *, size_t);
    std::string _filename;
    size_t _filesize = 0;
    EntryType _filetype = ENTRY_NONE;
    size_t _readoffset = 0;
    int64_t _current_header = 0;
    bool _eof = true; // to set enf of tar file as true
    void mark_end_of_file();
    void parse_current_header();
};