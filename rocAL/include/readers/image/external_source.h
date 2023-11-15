#include <vector>


class ExternalSourceImageReader {
   public:
    enum class Status {
        OK = 0
    };

    //! Feeds file names as an external_source into the reader
    virtual void feed_file_names(const std::vector<std::string> &file_names, size_t num_images, bool eos = false) = 0;

    //! Used for feeding raw data into the reader (mode specified compressed jpegs or raw)
    virtual void feed_data(const std::vector<unsigned char *> &images, const std::vector<size_t> &image_size, ExternalFileMode mode, bool eos = false, const std::vector<unsigned> roi_width = {}, const std::vector<unsigned> roi_height = {}, int width = 0, int height = 0, int channels = 0) = 0;

    virtual ~ExternalSourceImageReader() = default;
};
