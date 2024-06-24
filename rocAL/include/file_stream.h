
#include <string>
#include <memory>

class FileIOStream {
    public:
        static std::unique_ptr<FileIOStream> open(const std::string &path);
        void close_stream();
        size_t read_into_buffer(void * buffer, size_t read_n_bytes);
        const std::string& path() const { return _path; }
        size_t get_size() const;
        ptrdiff_t get_current_read_position() const;
        void set_read_position(ptrdiff_t pos, int whence = SEEK_SET);
        ~FileIOStream();
        FileIOStream(const std::string &path);

    private:
        FILE * _file_pointer;
        std::string _path;
};


