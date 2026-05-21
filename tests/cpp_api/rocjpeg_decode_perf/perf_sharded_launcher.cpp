#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

static bool is_jpeg(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext == ".jpg" || ext == ".jpeg";
}

static void usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " <dataset_dir> <num_gpus> <jpegdecodeperf_bin> [batch_size=32] [threads=4] [fmt=rgb] [work_dir=/tmp/rocjpeg_decode_perf/shards] [log_dir=/tmp/rocjpeg_decode_perf]\n"
        << "\n"
        << "Example:\n"
        << "  " << prog << " /path/to/images 8 /path/to/jpegdecodeperf\n";
}

int main(int argc, char** argv) {
    if (argc < 4) {
        usage(argv[0]);
        return 1;
    }

    const fs::path dataset_dir = argv[1];
    const int num_gpus = std::atoi(argv[2]);
    const fs::path jpegdecodeperf_bin = argv[3];
    const std::string batch_size = (argc > 4) ? argv[4] : "32";
    const std::string threads = (argc > 5) ? argv[5] : "4";
    const std::string fmt = (argc > 6) ? argv[6] : "rgb";
    const fs::path work_dir = (argc > 7) ? fs::path(argv[7]) : fs::path("/tmp/rocjpeg_decode_perf/shards");
    const fs::path log_dir = (argc > 8) ? fs::path(argv[8]) : fs::path("/tmp/rocjpeg_decode_perf");

    if (num_gpus <= 0) {
        std::cerr << "num_gpus must be > 0\n";
        return 1;
    }

    if (!fs::exists(dataset_dir)) {
        std::cerr << "Dataset does not exist: " << dataset_dir << "\n";
        return 1;
    }

    if (!fs::exists(jpegdecodeperf_bin)) {
        std::cerr << "jpegdecodeperf binary does not exist: " << jpegdecodeperf_bin << "\n";
        return 1;
    }

    std::vector<fs::path> files;
	const auto options = fs::directory_options::follow_directory_symlink;

	for (const auto& entry : fs::recursive_directory_iterator(dataset_dir, options)) {
		std::error_code ec;
		if (fs::is_regular_file(entry.status(ec)) && !ec && is_jpeg(entry.path())) {
			files.push_back(entry.path());
		}
	}

    std::sort(files.begin(), files.end());

    if (files.empty()) {
        std::cerr << "No JPEG files found under: " << dataset_dir << "\n";
        return 1;
    }

    fs::remove_all(work_dir);
    fs::create_directories(work_dir);
    fs::create_directories(log_dir);

    std::vector<int> shard_counts(num_gpus, 0);

    for (size_t i = 0; i < files.size(); ++i) {
        const int shard = static_cast<int>(i % static_cast<size_t>(num_gpus));
        const fs::path shard_dir = work_dir / ("shard_" + std::to_string(shard));
        fs::create_directories(shard_dir);

        const fs::path src = files[i];
        const std::string link_name =
            src.parent_path().filename().string() + "_" + src.filename().string();
        const fs::path dst = shard_dir / link_name;

        std::error_code ec;
        fs::create_symlink(src, dst, ec);
        if (ec) {
            std::cerr << "Failed to create symlink: " << dst << " -> " << src
                      << " error: " << ec.message() << "\n";
            return 1;
        }

        shard_counts[shard]++;
    }

    std::cout << "Total JPEG files: " << files.size() << "\n";
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        std::cout << "GPU " << gpu << " shard files: " << shard_counts[gpu] << "\n";
    }

    std::vector<pid_t> pids;

    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        const fs::path shard_dir = work_dir / ("shard_" + std::to_string(gpu));
        const fs::path log_path = log_dir / ("jpegdecodeperf_gpu" + std::to_string(gpu) + ".log");

        pid_t pid = fork();
        if (pid < 0) {
            std::cerr << "fork failed: " << std::strerror(errno) << "\n";
            return 1;
        }

        if (pid == 0) {
            FILE* log_file = std::freopen(log_path.c_str(), "w", stdout);
            if (!log_file) {
                std::perror("freopen stdout");
                _exit(127);
            }

            if (dup2(fileno(stdout), STDERR_FILENO) < 0) {
                std::perror("dup2 stderr");
                _exit(127);
            }

            const std::string gpu_id = std::to_string(gpu);

            execl(
                jpegdecodeperf_bin.c_str(),
                jpegdecodeperf_bin.c_str(),
                "-i", shard_dir.c_str(),
                "-b", batch_size.c_str(),
                "-t", threads.c_str(),
                "-fmt", fmt.c_str(),
                "-d", gpu_id.c_str(),
                static_cast<char*>(nullptr)
            );

            std::perror("execl jpegdecodeperf");
            _exit(127);
        }

        pids.push_back(pid);

        std::cout << "Launched GPU " << gpu << " pid " << pid
                  << " log " << log_path << "\n";
    }

    int failures = 0;

    for (pid_t pid : pids) {
        int status = 0;

        if (waitpid(pid, &status, 0) < 0) {
            std::cerr << "waitpid failed for pid " << pid << ": "
                      << std::strerror(errno) << "\n";
            failures++;
            continue;
        }

        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            std::cerr << "Process pid " << pid << " failed";

            if (WIFEXITED(status)) {
                std::cerr << " exit code " << WEXITSTATUS(status);
            } else if (WIFSIGNALED(status)) {
                std::cerr << " signal " << WTERMSIG(status);
            }

            std::cerr << "\n";
            failures++;
        }
    }

    if (failures) {
        std::cerr << failures << " jpegdecodeperf process(es) failed\n";
        return 1;
    }

    std::cout << "All jpegdecodeperf processes completed.\n";
    std::cout << "Logs: " << (log_dir / "jpegdecodeperf_gpu0.log")
              << " ... " << (log_dir / ("jpegdecodeperf_gpu" + std::to_string(num_gpus - 1) + ".log")) << "\n";

    return 0;
}
