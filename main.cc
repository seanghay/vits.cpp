#include <array>
#include <iostream>
#include <string>
#include "utf8.h"
#include <fstream>
#include <iostream>
#include "wavfile.hpp"
#include <onnxruntime/onnxruntime_cxx_api.h>

#define MODEL_FILENAME "G_96000.onnx"

const u_int32_t AUDIO_SAMPLE_RATE = 22050;
const float MAX_WAV_VALUE = 32767.0f;

int main(int argc, char *argv[])
{
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
  options.DisableCpuMemArena();
  options.DisableMemPattern();
  options.DisableProfiling();

  const Ort::Env environment = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "tts-app");
  Ort::Session session = Ort::Session(environment, MODEL_FILENAME, options);

  std::cout << "Model File loaded" << std::endl;

  auto memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  // inference
  std::vector<Ort::Value> inputTensors;

  // input
  std::vector<int64_t> input_values = {0, 22, 0, 32, 0, 12, 0, 6, 0, 22, 0, 7, 0, 36, 0, 6,
                                       0, 22, 0, 7, 0, 7, 0, 6, 0, 16, 0, 7, 0, 7, 0, 1,
                                       0, 25, 0, 13, 0, 36, 0, 6, 0, 23, 0, 14, 0, 13, 0, 9,
                                       0, 6, 0, 16, 0, 7, 0, 14, 0, 1, 0, 22, 0, 28, 0, 17,
                                       0, 6, 0, 23, 0, 32, 0, 8, 0, 1, 0, 31, 0, 9, 0, 9,
                                       0, 6, 0, 8, 0, 19, 0, 19, 0, 1, 0, 1, 0, 12, 0, 24,
                                       0, 18, 0, 1, 0, 1, 0, 22, 0, 7, 0, 9, 0, 18, 0, 1,
                                       0, 1, 0, 15, 0, 21, 0, 19, 0, 27, 0, 1, 0, 15, 0, 21,
                                       0, 28, 0, 6, 0, 8, 0, 9, 0, 12, 0, 1, 0, 8, 0, 28,
                                       0, 17, 0, 6, 0, 18, 0, 24, 0, 32, 0, 18, 0, 1, 0, 20,
                                       0, 21, 0, 7, 0, 17, 0, 6, 0, 20, 0, 13, 0, 13, 0, 1,
                                       0, 18, 0, 9, 0, 7, 0, 15, 0, 1, 0, 1, 0, 23, 0, 21,
                                       0, 32, 0, 25, 0, 6, 0, 29, 0, 7, 0, 7, 0, 18, 0, 1,
                                       0, 15, 0, 28, 0, 17, 0, 6, 0, 16, 0, 7, 0, 27, 0, 1,
                                       0, 18, 0, 30, 0, 6, 0, 15, 0, 30, 0, 30, 0, 6, 0, 29,
                                       0, 7, 0, 7, 0, 16, 0, 1, 0, 15, 0, 21, 0, 19, 0, 27,
                                       0, 1, 0, 18, 0, 19, 0, 7, 0, 17, 0, 1, 0, 15, 0, 16,
                                       0, 24, 0, 32, 0, 18, 0, 1, 0, 17, 0, 30, 0, 30, 0, 15,
                                       0, 1, 0, 36, 0, 28, 0, 20, 0, 6, 0, 21, 0, 24, 0, 17,
                                       0, 1, 0, 1, 0, 15, 0, 12, 0, 7, 0, 36, 0, 6, 0, 18,
                                       0, 7, 0, 36, 0, 1, 0, 20, 0, 24, 0, 32, 0, 15, 0, 6,
                                       0, 15, 0, 9, 0, 9, 0, 1, 0, 29, 0, 7, 0, 7, 0, 18,
                                       0, 1, 0, 31, 0, 19, 0, 23, 0, 1, 0, 20, 0, 12, 0, 7,
                                       0, 7, 0, 25, 0, 1, 0, 15, 0, 21, 0, 19, 0, 7, 0, 20,
                                       0, 6, 0, 29, 0, 7, 0, 9, 0, 15, 0, 1, 0, 15, 0, 18,
                                       0, 19, 0, 27, 0, 1, 0, 20, 0, 7, 0, 36, 0, 6, 0, 21,
                                       0, 13, 0, 36, 0, 6, 0, 25, 0, 9, 0, 9, 0, 18, 0, 1,
                                       0, 22, 0, 7, 0, 7, 0, 6, 0, 16, 0, 7, 0, 7, 0, 1,
                                       0, 1, 0, 29, 0, 28, 0, 27, 0, 6, 0, 15, 0, 28, 0, 28,
                                       0, 1, 0, 20, 0, 12, 0, 13, 0, 9, 0, 20, 0, 20, 0, 35,
                                       0, 9, 0, 7, 0, 15, 0, 6, 0, 20, 0, 36, 0, 7, 0, 32,
                                       0, 16, 0, 1, 0, 1, 0, 18, 0, 34, 0, 27, 0, 1, 0, 17,
                                       0, 13, 0, 9, 0, 18, 0, 1, 0, 15, 0, 7, 0, 7, 0, 6,
                                       0, 21, 0, 24, 0, 17, 0, 6, 0, 15, 0, 12, 0, 7, 0, 7,
                                       0, 18, 0, 1, 0, 31, 0, 28, 0, 16, 0, 1, 0, 15, 0, 7,
                                       0, 7, 0, 6, 0, 21, 0, 30, 0, 30, 0, 6, 0, 22, 0, 32,
                                       0, 15, 0, 6, 0, 22, 0, 7, 0, 7, 0, 1, 0, 1, 0, 18,
                                       0, 34, 0, 27, 0, 1, 0, 15, 0, 19, 0, 17, 0, 1, 0, 36,
                                       0, 7, 0, 19, 0, 14, 0, 1, 0, 22, 0, 32, 0, 12, 0, 1,
                                       0, 31, 0, 28, 0, 28, 0, 6, 0, 23, 0, 9, 0, 14, 0, 1,
                                       0, 23, 0, 13, 0, 32, 0, 23, 0, 1, 0, 23, 0, 25, 0, 32,
                                       0, 32, 0, 1, 0, 31, 0, 28, 0, 17, 0, 6, 0, 21, 0, 7,
                                       0, 20, 0, 1, 0, 23, 0, 7, 0, 7, 0, 17, 0, 4, 0};

  std::vector<int64_t> input_values_shape = {1, (int64_t)input_values.size()};
  inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(
      memoryInfo,
      input_values.data(),
      input_values.size(),
      input_values_shape.data(),
      input_values_shape.size()));

  // input_lengths
  std::vector<int64_t> input_lengths = {(int64_t)input_values.size()};
  std::vector<int64_t> input_lengths_shape = {(int64_t)input_lengths.size()};

  inputTensors.push_back(Ort::Value::CreateTensor(
      memoryInfo,
      input_lengths.data(), input_lengths.size(),
      input_lengths_shape.data(), input_lengths_shape.size()));

  // scales
  std::vector<float> scales = {0.667, 1, 0.8};
  std::vector<int64_t> scales_shape = {(int64_t)scales.size()};

  inputTensors.push_back(
      Ort::Value::CreateTensor(
          memoryInfo,
          scales.data(), scales.size(),
          scales_shape.data(), scales_shape.size()));

  std::array<const char *, 4> inputNames = {"input", "input_lengths", "scales", "sid"};
  std::array<const char *, 1> outputNames = {"output"};

  std::vector<Ort::Value> outputs = session.Run(
      Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(),
      inputTensors.size(), outputNames.data(), outputNames.size());

  const float *audio = outputs.front().GetTensorData<float>();
  auto audioShape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
  int64_t audioCount = audioShape[audioShape.size() - 1];
  std::cout << "audio duration: " << (double)audioCount / (double)22050 << " seconds" << std::endl;

  // normalize
  float maxAudioValue = 0.01f;
  for (int64_t i = 0; i < audioCount; i++)
  {
    float audioValue = abs(audio[i]);
    if (audioValue > maxAudioValue)
    {
      maxAudioValue = audioValue;
    }
  }

  // create audio buffer
  std::vector<int16_t> audioBuffer;
  audioBuffer.reserve(audioCount);

  float audioScale = (MAX_WAV_VALUE / std::max(0.01f, maxAudioValue));

  for (int64_t i = 0; i < audioCount; i++)
  {
    int16_t intAudioValue = static_cast<int16_t>(
        std::clamp(audio[i] * audioScale,
                   static_cast<float>(std::numeric_limits<int16_t>::min()),
                   static_cast<float>(std::numeric_limits<int16_t>::max())));
    audioBuffer.push_back(intAudioValue);
  }

  // Clean up
  for (std::size_t i = 0; i < outputs.size(); i++)
    Ort::detail::OrtRelease(outputs[i].release());
  
  for (std::size_t i = 0; i < inputTensors.size(); i++)
    Ort::detail::OrtRelease(inputTensors[i].release());

  std::ofstream audioFileStream("audio.wav", std::ios::binary);

  // write output wav file
  writeWavHeader(AUDIO_SAMPLE_RATE, 2, 1, sizeof(int16_t) * audioBuffer.size(), audioFileStream);
  audioFileStream.write((const char *)audioBuffer.data(),
                        sizeof(int16_t) * audioBuffer.size());

  audioFileStream.close();
}