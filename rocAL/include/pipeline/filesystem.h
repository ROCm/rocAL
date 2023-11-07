#ifndef FILESYSTEM_H
#define FILESYSTEM_H

#if defined(STD_FILESYSTEM_AVAILABLE)
    #include <filesystem>
    namespace filesys = std::filesystem;
#elif defined(EXPERIMENTAL_FILESYSTEM_AVAILABLE)
    #include <experimental/filesystem>
    namespace filesys = std::experimental::filesystem;
#else
    #error "No filesystem support available"
#endif

#endif // FILESYSTEM_H
