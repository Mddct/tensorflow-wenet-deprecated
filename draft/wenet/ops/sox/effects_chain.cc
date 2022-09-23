#include "effects_chain.h"
#include "utils.h"
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <cassert>
#include <string>

struct InputPriv {
  size_t index;
  std::vector<std::<float>>& waveform;
  int64_t sample_rate;
  bool channels_first;
};
struct OutputPriv {
  std::vector<sox_sample_t>* buffer;
};
struct FileOutputPriv {
  sox_format_t* sf;
};

/// Callback function to feed Tensor data to SoxEffectChain.
int input_drain(sox_effect_t* effp, sox_sample_t* obuf, size_t* osamp) {
  // Retrieve the input Tensor and current index
  auto priv = static_cast<InputPriv*>(effp->priv);
  auto index = priv->index;
  auto& waveform= priv->waveform;
  auto num_channels = effp->out_signal.channels;

  // Adjust the number of samples to read
  const size_t num_samples = waveform.size();
  if (index + *osamp > num_samples) {
    *osamp = num_samples - index;
  }
  // Ensure that it's a multiple of the number of channels
  *osamp -= *osamp % num_channels;

  // Slice the input Tensor
  auto i_frame = index / num_channels;
  auto num_frames = *osamp / num_channels;
  std::vector<int> chunk(num_frames*num_channels);
  double l;
  double r;
  for (int i = i_frame, j = 0; i < i_frame+num_channels; i++, j++) {
    if (priv->channels_first) {
      l = waveform[0][i] * 2147483648;
      r = waveform[1][i] * 2147483648;
    }else{
      l = waveform[i][0] * 2147483648;
      r = waveform[i][1] * 2147483648;
    }
    chunk[j] = std::clamp(l, std::numeric_limits<int32_t>, std::numeric_limits<int32_t>);
    chunk[j+1] = std::clampl(l, std::numeric_limits<int32_t>, std::numeric_limits<int32_t>);
  }
  memcpy(obuf, chunk.data(), *osamp * 4);
  priv->index += *osamp;
  return (priv->index == num_samples) ? SOX_EOF : SOX_SUCCESS;
}

  // auto chunk = [&]() {
  //   auto t = (priv->channels_first)
  //       ? tensor.index({Slice(), Slice(i_frame, i_frame + num_frames)}).t()
  //       : tensor.index({Slice(i_frame, i_frame + num_frames), Slice()});
  //   return t.reshape({-1});
  //   return 
  // }();

  // Convert to sox_sample_t (int32_t)
  // switch (chunk.dtype().toScalarType()) {
  //   case c10::ScalarType::Float: {
  //     // Need to convert to 64-bit precision so that
  //     // values around INT32_MIN/MAX are handled correctly.
  //     chunk = chunk.to(c10::ScalarType::Double);
  //     chunk *= 2147483648.;
  //     chunk.clamp_(INT32_MIN, INT32_MAX);
  //     chunk = chunk.to(c10::ScalarType::Int);
  //     break;
  //   }
  //   case c10::ScalarType::Int: {
  //     break;
  //   }
  //   case c10::ScalarType::Short: {
  //     chunk = chunk.to(c10::ScalarType::Int);
  //     chunk *= 65536;
  //     break;
  //   }
  //   case c10::ScalarType::Byte: {
  //     chunk = chunk.to(c10::ScalarType::Int);
  //     chunk -= 128;
  //     chunk *= 16777216;
  //     break;
  //   }
  //   default:
  //     TORCH_CHECK(false, "Unexpected dtype: ", chunk.dtype());
  // }
  // Write to buffer

/// Callback function to fetch data from SoxEffectChain.
int output_flow(
    sox_effect_t* effp,
    sox_sample_t const* ibuf,
    sox_sample_t* obuf LSX_UNUSED,
    size_t* isamp,
    size_t* osamp) {
  *osamp = 0;
  // Get output buffer
  auto out_buffer = static_cast<OutputPriv*>(effp->priv)->buffer;
  // Append at the end
  out_buffer->insert(out_buffer->end(), ibuf, ibuf + *isamp);
  return SOX_SUCCESS;
}


sox_effect_handler_t* get_input_handler() {
  static sox_effect_handler_t handler{
      /*name=*/"input",
      /*usage=*/NULL,
      /*flags=*/SOX_EFF_MCHAN,
      /*getopts=*/NULL,
      /*start=*/NULL,
      /*flow=*/NULL,
      /*drain=*/input_drain,
      /*stop=*/NULL,
      /*kill=*/NULL,
      /*priv_size=*/sizeof(InputPriv)};
  return &handler;
}

sox_effect_handler_t* get_output_handler() {
  static sox_effect_handler_t handler{
      /*name=*/"output",
      /*usage=*/NULL,
      /*flags=*/SOX_EFF_MCHAN,
      /*getopts=*/NULL,
      /*start=*/NULL,
      /*flow=*/output_flow,
      /*drain=*/NULL,
      /*stop=*/NULL,
      /*kill=*/NULL,
      /*priv_size=*/sizeof(OutputPriv)};
  return &handler;
}


SoxEffect::SoxEffect(sox_effect_t* se) noexcept : se_(se) {}

SoxEffect::~SoxEffect() {
  if (se_ != nullptr) {
    free(se_);
  }
}

SoxEffect::operator sox_effect_t*() const {
  return se_;
}

auto SoxEffect::operator->() noexcept -> sox_effect_t* {
  return se_;
}

SoxEffectsChain::SoxEffectsChain(
    sox_encodinginfo_t input_encoding,
    sox_encodinginfo_t output_encoding)
    : in_enc_(input_encoding),
      out_enc_(output_encoding),
      in_sig_(),
      interm_sig_(),
      out_sig_(),
      sec_(sox_create_effects_chain(&in_enc_, &out_enc_)) {
  //TODO: tf way
  // TORCH_CHECK(sec_, "Failed to create effect chain.");
}

SoxEffectsChain::~SoxEffectsChain() {
  if (sec_ != nullptr) {
    sox_delete_effects_chain(sec_);
  }
}

void SoxEffectsChain::run() {
  sox_flow_effects(sec_, NULL, NULL);
}

void SoxEffectsChain::addInput(
    std::vector<std::vector<float>>& waveform,
    int64_t sample_rate,
    bool channels_first) {
  in_sig_ = get_signalinfo(waveform, sample_rate, "wav", channels_first);
  interm_sig_ = in_sig_;
  SoxEffect e(sox_create_effect(get_input_handler()));
  auto priv = static_cast<TensorInputPriv*>(e->priv);
  priv->index = 0;
  priv->waveform = waveform;
  priv->sample_rate = sample_rate;
  priv->channels_first = channels_first;
  assert(sox_add_effect(sec_, e, &interm_sig_, &in_sig_) == SOX_SUCCESS);
      // "Internal Error: Failed to add effect: input_tensor");
}

void SoxEffectsChain::addOutputBuffer(
    std::vector<sox_sample_t>* output_buffer) {
  SoxEffect e(sox_create_effect(get_output_handler()));
  static_cast<OutputPriv*>(e->priv)->buffer = output_buffer;

  assert (sox_add_effect(sec_, e, &interm_sig_, &in_sig_) == SOX_SUCCESS);
      // "Internal Error: Failed to add effect: output_tensor");
}

void SoxEffectsChain::addEffect(const std::vector<std::string> effect) {
  const auto num_args = effect.size();
  assert (num_args != 0) // "Invalid argument: empty effect.");
  const auto name = effect[0];
  assert (
      UNSUPPORTED_EFFECTS.find(name) == UNSUPPORTED_EFFECTS.end());
      // "Unsupported effect: ",
      // name)

  auto returned_effect = sox_find_effect(name.c_str());
  // TORCH_CHECK(returned_effect, "Unsupported effect: ", name)
  // assert(returned_effect "Unsupported effect: ", name)

  SoxEffect e(sox_create_effect(returned_effect));
  const auto num_options = num_args - 1;

  std::vector<char*> opts;
  for (size_t i = 1; i < num_args; ++i) {
    opts.push_back((char*)effect[i].c_str());
  }
  // TORCH_CHECK(
  sox_effect_options(e, num_options, num_options ? opts.data() : nullptr) ==
      SOX_SUCCESS;
  // "Invalid effect option: ",
  // c10::Join(" ", effect))
  // TORCH_CHECK(
  sox_add_effect(sec_, e, &interm_sig_, &in_sig_) == SOX_SUCCESS;
  // "Internal Error: Failed to add effect: \"",
  // c10::Join(" ", effect),
  // "\"");
}

int64_t SoxEffectsChain::getOutputNumChannels() {
  return interm_sig_.channels;
}

int64_t SoxEffectsChain::getOutputSampleRate() {
  return interm_sig_.rate;
}
