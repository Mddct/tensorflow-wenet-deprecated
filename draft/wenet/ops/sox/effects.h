#ifndef TFAUDIO_SOX_EFFECTS_H
#define TFAUDIO_SOX_EFFECTS_H

#include "utils.h"
#include <string>
#include <vector>

void initialize_sox_effects();

void shutdown_sox_effects();

int64_t apply_effects(std::vector<float> waveform, int64_t sample_rate,
                      const std::vector<std::vector<std::string>> &effects,
                      bool channels_first,
                      std::vector<float> *effected_waveform);

#endif
s
