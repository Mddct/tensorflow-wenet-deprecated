#ifndef TFAUDIO_SOX_UTILS_H
#define TFAUDIO_SOX_UTILS_H

#include "sox.h"
#include <unordered_set>
#include <vector>
#include <string>

////////////////////////////////////////////////////////////////////////////////
// APIs for Python interaction
////////////////////////////////////////////////////////////////////////////////

/// Set sox global options
void set_seed(const int64_t seed);

void set_verbosity(const int64_t verbosity);

void set_use_threads(const bool use_threads);

void set_buffer_size(const int64_t buffer_size);

int64_t get_buffer_size();

std::vector<std::vector<std::string>> list_effects();

std::vector<std::string> list_read_formats();

std::vector<std::string> list_write_formats();

////////////////////////////////////////////////////////////////////////////////
// Utilities for sox_io / sox_effects implementations
////////////////////////////////////////////////////////////////////////////////

const std::unordered_set<std::string> UNSUPPORTED_EFFECTS =
    {"input", "output", "spectrogram", "noiseprof", "noisered", "splice"};

/// helper class to automatically close sox_format_t*
struct SoxFormat {
  explicit SoxFormat(sox_format_t* fd) noexcept;
  SoxFormat(const SoxFormat& other) = delete;
  SoxFormat(SoxFormat&& other) = delete;
  SoxFormat& operator=(const SoxFormat& other) = delete;
  SoxFormat& operator=(SoxFormat&& other) = delete;
  ~SoxFormat();
  sox_format_t* operator->() const noexcept;
  operator sox_format_t*() const noexcept;

  void close();

 private:
  sox_format_t* fd_;
};

/// Extract extension from file path
const std::string get_filetype(const std::string path);

sox_signalinfo_t get_signalinfo(const std::vector<std::vector<float>> &waveform,
                                const int64_t sample_rate,
                                const std::string filetype,
                                const bool channels_first); 

#endif
