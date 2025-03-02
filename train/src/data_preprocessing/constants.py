# Copyright © 2023 Arm Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

wanted_words = ["yes","no","up","down","left","right","on","off","stop","go"]
background_frequency=0.8
background_volume=0.1
batch_size=100
clip_duration_ms=1000
data_dir='./data/'
data_url='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
dct_coefficient_count=10
eval_step_interval=400
how_many_training_steps='10000,10000,10000'
model_architecture='ds_cnn'
model_size_info=[5, 64, 10, 4, 2, 2, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1]
sample_rate=16000
silence_percentage=10.0
testing_percentage=10
time_shift_ms=100.0
unknown_percentage=10.0
validation_percentage=10
# wanted_words='yes,no,up,down,left,right,on,off,stop,go'
window_size_ms=40.0
window_stride_ms=20.

def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
    """Calculates common settings needed for all models.

    Args:
        label_count: How many classes are to be recognized.
        sample_rate: Number of audio samples per second. Default: 16Hz=16000
        clip_duration_ms: Length of each audio clip to be analyzed. Default: 1000ms=1s
        window_size_ms: Duration of frequency analysis window. Default: 40ms
        window_stride_ms: How far to move in time between frequency windows. Default: 20ms
        dct_coefficient_count: Number of frequency bins to use for analysis. Default: 10

    Returns:
        Dictionary containing common settings.
    """
    desired_samples = int(sample_rate * clip_duration_ms / 1000) # 16000 * 1000 / 1000 = 16000
    window_size_samples = int(sample_rate * window_size_ms / 1000) # 16000 * 40 / 1000 = 640
    window_stride_samples = int(sample_rate * window_stride_ms / 1000) # 16000 * 20 / 1000 = 320
    length_minus_window = (desired_samples - window_size_samples) # 16000 - 640 = 15360
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples) # 1 + 15360 / 320 = 49
    fingerprint_size = dct_coefficient_count * spectrogram_length # 10 * 49 = 490

    return {
        'desired_samples': desired_samples, # 16000
        'window_size_samples': window_size_samples, # 640
        'window_stride_samples': window_stride_samples, # 320
        'spectrogram_length': spectrogram_length, # 49
        'dct_coefficient_count': dct_coefficient_count, # 10
        'fingerprint_size': fingerprint_size, # 490
        'label_count': label_count, 
        'sample_rate': sample_rate, # 16000
    }