
#include <string>
#include <memory>

class StdFileStream {
    public:
        static std::unique_ptr<StdFileStream> Open(const std::string &uri);
        void Close();
        size_t Read(void * buffer, size_t n_bytes);
        const std::string& path() const { return _path; }
        size_t Size() const;
        ptrdiff_t TellRead() const;
        void SeekRead(ptrdiff_t pos, int whence = SEEK_SET);
        ~StdFileStream();
        StdFileStream(const std::string &path);

    private:
        FILE * fp_;
        std::string _path;
};


