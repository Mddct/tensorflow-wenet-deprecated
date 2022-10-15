#include "wenet/tfaudio/cc/ops/kaldi_fbank_kernels.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <utility>
#include <vector>
#include <cassert>

// This code copy from wenet
namespace wenet {

#ifndef M_2PI
#define M_2PI 6.283185307179586476925286766559005
#endif
void make_sintbl(int n, float *sintbl) {
  int i, n2, n4, n8;
  float c, s, dc, ds, t;

  n2 = n / 2;
  n4 = n / 4;
  n8 = n / 8;
  t = sin(M_PI / n);
  dc = 2 * t * t;
  ds = sqrt(dc * (2 - dc));
  t = 2 * dc;
  c = sintbl[n4] = 1;
  s = sintbl[0] = 0;
  for (i = 1; i < n8; ++i) {
    c -= dc;
    dc += t * c;
    s += ds;
    ds -= t * s;
    sintbl[i] = s;
    sintbl[n4 - i] = c;
  }
  if (n8 != 0)
    sintbl[n8] = sqrt(0.5);
  for (i = 0; i < n4; ++i)
    sintbl[n2 - i] = sintbl[i];
  for (i = 0; i < n2 + n4; ++i)
    sintbl[i + n2] = -sintbl[i];
}

void make_bitrev(int n, int *bitrev) {
  int i, j, k, n2;

  n2 = n / 2;
  i = j = 0;
  for (;;) {
    bitrev[i] = j;
    if (++i >= n)
      break;
    k = n2;
    while (k <= j) {
      j -= k;
      k /= 2;
    }
    j += k;
  }
}

// bitrev: bit reversal table
// sintbl: trigonometric function table
// x:real part
// y:image part
// n: fft length
int fft(const int *bitrev, const float *sintbl, float *x, float *y, int n) {
  int i, j, k, ik, h, d, k2, n4, inverse;
  float t, s, c, dx, dy;

  /* preparation */
  if (n < 0) {
    n = -n;
    inverse = 1; /* inverse transform */
  } else {
    inverse = 0;
  }
  n4 = n / 4;
  if (n == 0) {
    return 0;
  }

  /* bit reversal */
  for (i = 0; i < n; ++i) {
    j = bitrev[i];
    if (i < j) {
      t = x[i];
      x[i] = x[j];
      x[j] = t;
      t = y[i];
      y[i] = y[j];
      y[j] = t;
    }
  }

  /* transformation */
  for (k = 1; k < n; k = k2) {
    h = 0;
    k2 = k + k;
    d = n / k2;
    for (j = 0; j < k; ++j) {
      c = sintbl[h + n4];
      if (inverse)
        s = -sintbl[h];
      else
        s = sintbl[h];
      for (i = j; i < n; i += k2) {
        ik = i + k;
        dx = s * y[ik] + c * x[ik];
        dy = c * y[ik] - s * x[ik];
        x[ik] = x[i] - dx;
        x[i] += dx;
        y[ik] = y[i] - dy;
        y[i] += dy;
      }
      h += d;
    }
  }
  if (inverse) {
    /* divide by n in case of the inverse transformation */
    for (i = 0; i < n; ++i) {
      x[i] /= n;
      y[i] /= n;
    }
  }
  return 0; /* finished successfully */
}

// This code is based on kaldi Fbank implementation, please see
// https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-fbank.cc
class Fbank {
public:
  Fbank(int num_bins, int sample_rate, int frame_length, int frame_shift, float dither)
      : num_bins_(num_bins), sample_rate_(sample_rate),
        frame_length_(frame_length), frame_shift_(frame_shift), use_log_(true),
        remove_dc_offset_(true), generator_(0), distribution_(0, 1.0),
        dither_(dither) {
    fft_points_ = UpperPowerOfTwo(frame_length_);
    // generate bit reversal table and trigonometric function table
    const int fft_points_4 = fft_points_ / 4;
    bitrev_.resize(fft_points_);
    sintbl_.resize(fft_points_ + fft_points_4);
    make_sintbl(fft_points_, sintbl_.data());
    make_bitrev(fft_points_, bitrev_.data());

    int num_fft_bins = fft_points_ / 2;
    float fft_bin_width = static_cast<float>(sample_rate_) / fft_points_;
    int low_freq = 20, high_freq = sample_rate_ / 2;
    float mel_low_freq = MelScale(low_freq);
    float mel_high_freq = MelScale(high_freq);
    float mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1);
    bins_.resize(num_bins_);
    center_freqs_.resize(num_bins_);
    for (int bin = 0; bin < num_bins; ++bin) {
      float left_mel = mel_low_freq + bin * mel_freq_delta,
            center_mel = mel_low_freq + (bin + 1) * mel_freq_delta,
            right_mel = mel_low_freq + (bin + 2) * mel_freq_delta;
      center_freqs_[bin] = InverseMelScale(center_mel);
      std::vector<float> this_bin(num_fft_bins);
      int first_index = -1, last_index = -1;
      for (int i = 0; i < num_fft_bins; ++i) {
        float freq = (fft_bin_width * i); // Center frequency of this fft
        // bin.
        float mel = MelScale(freq);
        if (mel > left_mel && mel < right_mel) {
          float weight;
          if (mel <= center_mel)
            weight = (mel - left_mel) / (center_mel - left_mel);
          else
            weight = (right_mel - mel) / (right_mel - center_mel);
          this_bin[i] = weight;
          if (first_index == -1)
            first_index = i;
          last_index = i;
        }
      }
      assert (first_index != -1 && last_index >= first_index);
      bins_[bin].first = first_index;
      int size = last_index + 1 - first_index;
      bins_[bin].second.resize(size);
      for (int i = 0; i < size; ++i) {
        bins_[bin].second[i] = this_bin[first_index + i];
      }
    }

    // povey window
    povey_window_.resize(frame_length_);
    double a = M_2PI / (frame_length - 1);
    for (int i = 0; i < frame_length; ++i) {
      povey_window_[i] = pow(0.5 - 0.5 * cos(a * i), 0.85);
    }
  }

  void set_use_log(bool use_log) { use_log_ = use_log; }

  void set_remove_dc_offset(bool remove_dc_offset) {
    remove_dc_offset_ = remove_dc_offset;
  }

  void set_dither(float dither) { dither_ = dither; }

  int num_bins() const { return num_bins_; }

  static inline float InverseMelScale(float mel_freq) {
    return 700.0f * (expf(mel_freq / 1127.0f) - 1.0f);
  }

  static inline float MelScale(float freq) {
    return 1127.0f * logf(1.0f + freq / 700.0f);
  }

  static int UpperPowerOfTwo(int n) {
    return static_cast<int>(pow(2, ceil(log(n) / log(2))));
  }

  // pre emphasis
  void PreEmphasis(float coeff, std::vector<float> *data) const {
    if (coeff == 0.0)
      return;
    for (int i = data->size() - 1; i > 0; i--)
      (*data)[i] -= coeff * (*data)[i - 1];
    (*data)[0] -= coeff * (*data)[0];
  }

  // Apply povey window on data in place
  void Povey(std::vector<float> *data) const {
    //TODO: replace with tf
    assert (data->size() >= povey_window_.size());
    for (size_t i = 0; i < povey_window_.size(); ++i) {
      (*data)[i] *= povey_window_[i];
    }
  }

  // Compute fbank feat, return num frames
  int Compute(const std::vector<float> &wave,
              std::vector<std::vector<float>> *feat) {
    int num_samples = wave.size();
    if (num_samples < frame_length_)
      return 0;
    int num_frames = 1 + ((num_samples - frame_length_) / frame_shift_);
    feat->resize(num_frames);
    std::vector<float> fft_real(fft_points_, 0), fft_img(fft_points_, 0);
    std::vector<float> power(fft_points_ / 2);
    for (int i = 0; i < num_frames; ++i) {
      std::vector<float> data(wave.data() + i * frame_shift_,
                              wave.data() + i * frame_shift_ + frame_length_);
      // optional add noise
      if (dither_ != 0.0) {
        for (size_t j = 0; j < data.size(); ++j)
          data[j] += dither_ * distribution_(generator_);
      }
      // optinal remove dc offset
      if (remove_dc_offset_) {
        float mean = 0.0;
        for (size_t j = 0; j < data.size(); ++j)
          mean += data[j];
        mean /= data.size();
        for (size_t j = 0; j < data.size(); ++j)
          data[j] -= mean;
      }

      PreEmphasis(0.97, &data);
      Povey(&data);
      // copy data to fft_real
      memset(fft_img.data(), 0, sizeof(float) * fft_points_);
      memset(fft_real.data() + frame_length_, 0,
             sizeof(float) * (fft_points_ - frame_length_));
      memcpy(fft_real.data(), data.data(), sizeof(float) * frame_length_);
      fft(bitrev_.data(), sintbl_.data(), fft_real.data(), fft_img.data(),
          fft_points_);
      // power
      for (int j = 0; j < fft_points_ / 2; ++j) {
        power[j] = fft_real[j] * fft_real[j] + fft_img[j] * fft_img[j];
      }

      (*feat)[i].resize(num_bins_);
      // cepstral coefficients, triangle filter array
      for (int j = 0; j < num_bins_; ++j) {
        float mel_energy = 0.0;
        int s = bins_[j].first;
        for (size_t k = 0; k < bins_[j].second.size(); ++k) {
          mel_energy += bins_[j].second[k] * power[s + k];
        }
        // optional use log
        if (use_log_) {
          if (mel_energy < std::numeric_limits<float>::epsilon())
            mel_energy = std::numeric_limits<float>::epsilon();
          mel_energy = logf(mel_energy);
        }

        (*feat)[i][j] = mel_energy;
      }
    }
    return num_frames;
  }

private:
  int num_bins_;
  int sample_rate_;
  int frame_length_, frame_shift_;
  int fft_points_;
  bool use_log_;
  bool remove_dc_offset_;
  std::vector<float> center_freqs_;
  std::vector<std::pair<int, std::vector<float>>> bins_;
  std::vector<float> povey_window_;
  std::default_random_engine generator_;
  std::normal_distribution<float> distribution_;
  float dither_;

  // bit reversal table
  std::vector<int> bitrev_;
  // trigonometric function table
  std::vector<float> sintbl_;
};

} // namespace wenet

using namespace tensorflow;

REGISTER_OP("KaldiFbankOp")
    .Input("samples: float32")
    .Attr("sample_rate: int")
    .Attr("frame_length: int")
    .Attr("frame_shift: int")
    .Attr("bins: int")
    .Attr("dither: float")
    .Output("fbank: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      // TODO: check shape later
      return Status::OK();
    });

class KaldiFbankOp : public OpKernel {
public:
  explicit KaldiFbankOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sample_rate", &sample_rate_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("frame_length", &frame_length_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("frame_shift", &frame_shift_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bins", &bins_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dither", &dither_));

    frame_length_ = sample_rate_ / 1000 * frame_length_;
    frame_shift_ = sample_rate_ / 1000 * 10;
  }

  void Compute(OpKernelContext* ctx) override {

    // Grab the input tensor
    const Tensor& input_tensor = ctx->input(0);
    auto*  input_ptr = input_tensor.flat<float_t>().data();
    std::vector<float_t> samples(input_ptr, input_tensor.NumElements()+input_ptr);

    wenet::Fbank fbank(bins_, sample_rate_, frame_length_, frame_shift_, dither_);

    std::vector<std::vector<float>> feat;
    int num_frames = fbank.Compute(samples, &feat);
    // Create an output tensor
    Tensor *output_tensor = NULL;
    TensorShape output_shape;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
        0, TensorShape({num_frames, bins_}), &output_tensor));
    auto output_flat = output_tensor->flat<float_t>();

    // Set all but the first element of the output tensor to 0.
    int N = output_flat.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = feat[int(i/bins_)][i%bins_];
    }
  }
 private:
  int sample_rate_;
  int frame_length_;
  int frame_shift_;
  int bins_; 
  float dither_;
};

REGISTER_KERNEL_BUILDER(Name("KaldiFbankOp").Device(DEVICE_CPU),KaldiFbankOp);

