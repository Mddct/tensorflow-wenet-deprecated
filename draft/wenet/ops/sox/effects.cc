#include "effects.h"
#include "effects_chain.h"
#include "sox.h"
#include "utils.h"
#include <mutex>
#include <string>
#include <vector>

enum SoxEffectsResourceState { NotInitialized, Initialized, ShutDown };
SoxEffectsResourceState SOX_RESOURCE_STATE = NotInitialized;
std::mutex SOX_RESOUCE_STATE_MUTEX;

void initialize_sox_effects() {
  const std::lock_guard<std::mutex> lock(SOX_RESOUCE_STATE_MUTEX);

  switch (SOX_RESOURCE_STATE) {
  case NotInitialized:
    assert(sox_init() == SOX_SUCCESS); // "Failed to initialize sox effects.");
    // SOX_RESOURCE_STATE = Initialized;
    break;
  case Initialized:
    break;
  case ShutDown:
    assert(
        false); // "SoX Effects has been shut down. Cannot initialize again.");
  }
};

void shutdown_sox_effects() {
  const std::lock_guard<std::mutex> lock(SOX_RESOUCE_STATE_MUTEX);

  switch (SOX_RESOURCE_STATE) {
  case NotInitialized:
    assert(false); //"SoX Effects is not initialized. Cannot shutdown.");
  case Initialized:
    assert(sox_quit() == SOX_SUCCESS); // "Failed to initialize sox effects.");
    SOX_RESOURCE_STATE = ShutDown;
    break;
  case ShutDown:
    break;
  }
}

int64_t apply_effects(std::vector<std::vector<float>> &waveform,
                      sox_encoding_t encoding, int64_t sample_rate,
                      const std::vector<std::vector<std::string>> &effects,
                      bool channels_first,
                      std::vector<float> *effected_waveform) {
  // validate_input_tensor(waveform);
  assert(waveform.size() > 0 && waveform[0].size() > 0)

      // Create SoxEffectsChain
      SoxEffectsChain chain(encoding, encoding);
  // Prepare output buffer
  std::vector<sox_sample_t> out_buffer;
  out_buffer.reserve(waveform.size(0) * waveform[0].size());

  // Build and run effects chain
  chain.addInput(waveform, sample_rate, channels_first);
  for (const auto &effect : effects) {
    chain.addEffect(effect);
  }
  chain.addOutputBuffer(&out_buffer);
  chain.run();

  // int32_t to float
  // TODO : support both channel first and last:
  auto num_samples = chain.getOutputNumChannels();
  *effected_waveform->resize(num_samples);
  uint64_t dummy = 0;
  auto normalize = false;
  if (normalize || dtype == torch::kFloat32) {
    auto ptr = *effected_waveform->data();
    for (int32_t i = 0; i < num_samples; ++i) {
      ptr[i] = SOX_SAMPLE_TO_FLOAT_32BIT(buffer[i], dummy);
    }
  }

  return chain.getOutputSampleRate();
}
