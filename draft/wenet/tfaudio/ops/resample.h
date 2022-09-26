#ifndef TFAUDIO_OPS_KERNELS_RESAMPLE_H_
#define TFAUDIO_OPS_KERNELS_RESAMPLE_H_

#include <stdlib.h>
#include <vector>
#include <assert.h>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;  // NOLINT


class ArbitraryResample {
 public:
  ArbitraryResample(int num_samples_in,
                    float samp_rate_hz,
                    float filter_cutoff_hz,
                    const std::vector<float> &sample_points_secs,
                    int num_zeros);

  int NumSamplesIn() const { return num_samples_in_; }

  int NumSamplesOut() const { return weights_.size(); }
  void Resample(const std::vector<std::vector<float> > &input,
                std::vector<std::vector<float> > *output) const;

  void Resample(const std::vector<float> &input,
                std::vector<float> *output) const;
 private:
  void SetIndexes(const std::vector<float> &sample_points);

  void SetWeights(const std::vector<float> &sample_points);

  float FilterFunc(float t) const;

  int num_samples_in_;
  float samp_rate_in_;
  float filter_cutoff_;
  int num_zeros_;

  std::vector<int> first_index_;
  std::vector<std::vector<float> > weights_;
};

class LinearResample {
 public:
  LinearResample(int samp_rate_in_hz,
                 int samp_rate_out_hz,
                 float filter_cutoff_hz,
                 int num_zeros);

  void Resample(const std::vector<float> &input,
                bool flush,
                std::vector<float> *output);

  void Reset();

  //// Return the input and output sampling rates (for checks, for example)
  inline int GetInputSamplingRate() { return samp_rate_in_; }
  inline int GetOutputSamplingRate() { return samp_rate_out_; }
 private:
  int GetNumOutputSamples(int input_num_samp, bool flush) const;

  inline void GetIndexes(int samp_out,
                         int *first_samp_in,
                         int *samp_out_wrapped) const;

  void SetRemainder(const std::vector<float> &input);

  void SetIndexesAndWeights();

  float FilterFunc(float) const;

  // The following variables are provided by the user.
  int samp_rate_in_;
  int samp_rate_out_;
  float filter_cutoff_;
  int num_zeros_;

  int input_samples_in_unit_;
  int output_samples_in_unit_;

  std::vector<int> first_index_;
  std::vector<std::vector<float> > weights_;

  int input_sample_offset_;
  int output_sample_offset_;
  std::vector<float> input_remainder_;
};

void ResampleWaveform(float orig_freq, const std::vector<float> &wave,
                      float new_freq, std::vector<float> *new_wave);

inline void DownsampleWaveForm(float orig_freq, const std::vector<float> &wave,
                               float new_freq, std::vector<float> *new_wave) {
  ResampleWaveform(orig_freq, wave, new_freq, new_wave);
}

#endif  // TFAUDIO_OPS_KERNELS_RESAMPLE_H_
