// ============================================================================
// Cross-platform Filesystem Include Header
// ============================================================================
// This header provides cross-platform filesystem support for different compilers
// and operating systems.

#ifndef FILESYSTEM_INCLUDE_H
#define FILESYSTEM_INCLUDE_H

// Cross-platform filesystem include
#ifndef __GNUC__
    #include <filesystem>
    namespace fs = std::filesystem;
#else
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
#endif

#endif // FILESYSTEM_INCLUDE_H