from data_preprocessing import data_preprocessing
from data_preprocessing import constants
import tensorflow as tf
from . import Dataset

word_list = data_preprocessing.prepare_words_list(wanted_words=constants.wanted_words)
print(word_list)

model_settings = constants.prepare_model_settings(
    len(data_preprocessing.prepare_words_list(constants.wanted_words)),
    constants.sample_rate,
    constants.clip_duration_ms,
    constants.window_size_ms,
    constants.window_stride_ms,
    constants.dct_coefficient_count,
)


class Speech(Dataset):
    def __init__(self):
        self.audio_processor = data_preprocessing.AudioProcessor(
            data_url=constants.data_url,
            data_dir=constants.data_dir,
            silence_percentage=constants.silence_percentage,
            unknown_percentage=constants.unknown_percentage,
            wanted_words=constants.wanted_words,
            validation_percentage=constants.validation_percentage,
            testing_percentage=constants.testing_percentage,
            model_settings=model_settings,
        )
        self.train_data: tf.data.Dataset = self.audio_processor.get_data(
            mode=self.audio_processor.Modes.TRAINING,
            background_frequency=constants.background_frequency,
            background_volume_range=constants.background_volume,
            time_shift=int((constants.time_shift_ms * constants.sample_rate) / 1000),
        )
        self.val_data: tf.data.Dataset = self.audio_processor.get_data(
            mode=self.audio_processor.Modes.VALIDATION
        )
        self.test_data: tf.data.Dataset = self.audio_processor.get_data(
            mode=self.audio_processor.Modes.TESTING
        )
        self.__preprocess__()

    def get_data(self):
        return self.train_data, self.val_data, self.test_data

    def get_scaled_data(self):
        pass

    def representative_data_gen(self):
        for mfcc, label in self.val_data.take(100):
            yield [mfcc]

    def get_audio_processor(self):
        return self.audio_processor

    def __preprocess__(self):
        self.train_data = (
            self.train_data.repeat()
            .batch(int(constants.batch_size))
            .prefetch(tf.data.AUTOTUNE)
        )
        self.val_data = self.val_data.batch(int(constants.batch_size)).prefetch(
            tf.data.AUTOTUNE
        )
        self.test_data = self.test_data.batch(int(constants.batch_size)).prefetch(
            tf.data.AUTOTUNE
        )

SpeechCommand = Speech()