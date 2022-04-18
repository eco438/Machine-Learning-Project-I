import tensorflow as tf
import numpy as np
import librosa
from tqdm.notebook import tqdm
from tensorflow.keras.utils import to_categorical
import math
import sys
rawtrain = tf.data.TFRecordDataset("nsynth-train.tfrecord")
rawvalid = tf.data.TFRecordDataset("nsynth-valid.tfrecord")
rawtest = tf.data.TFRecordDataset("nsynth-test.tfrecord")


def _parseme(raw_audio_record):
	feature_description = {
	    'note': tf.io.FixedLenFeature([], tf.int64, default_value=0),
	    'note_str': tf.io.FixedLenFeature([], tf.string, default_value=''),
	    'instrument': tf.io.FixedLenFeature([], tf.int64, default_value=0),
	    'instrument_str': tf.io.FixedLenFeature([], tf.string, default_value=''),
	    'pitch': tf.io.FixedLenFeature([], tf.int64, default_value=0),
	    'velocity': tf.io.FixedLenFeature([], tf.int64,default_value=0),
	    'sample_rate': tf.io.FixedLenFeature([], tf.int64, default_value=0),
	    'audio': tf.io.FixedLenSequenceFeature([], tf.float32,  allow_missing=True, default_value=0.0),
	    'qualities': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
	    'qualities_str': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True, default_value=''),
	    'instrument_family': tf.io.FixedLenFeature([], tf.int64, default_value=0),
	    'instrument_family_str': tf.io.FixedLenFeature([], tf.string, default_value=''),
	    'instrument_source': tf.io.FixedLenFeature([], tf.int64, default_value=0),
	    'instrument_source_str': tf.io.FixedLenFeature([], tf.string, default_value='')     
	}

	return tf.io.parse_single_example(raw_audio_record, feature_description)


# Converts a single value in the range -1, 1 to its mel frequency equivalent
def wavtomel(audio):
	hz = audio * 32767
	mel = 1127 * math.log(1 + hz/700)
	return mel


# Call tf.map_fn(wavtomelwrap, t) to apply wavtomel on every float in Tensor t.
def wavtomelwrap(x):
	return tf.cast(tf.py_function(wavtomel, [x], tf.float64), tf.float32)


datatrain = rawtrain.map(_parseme)
datatrain = datatrain.filter(lambda d: int(d["instrument_family"]) != 9)  # remove synth lead as instructed.
# This is all the audio frequencies converted to mel, our input to learn
dataytrain = datatrain.map(lambda d: d["instrument_family"])
# This is all of the instrument family indices, our output to learn
# Index meaning can be found at https://magenta.tensorflow.org/datasets/nsynth#instrument-families

datavalid = rawvalid.map(_parseme)
datavalid = datavalid.filter(lambda d: int(d["instrument_family"]) != 9)  # remove synth lead as instructed.
# This is all the audio frequencies converted to mel, our input to try
datayvalid = datavalid.map(lambda d: d["instrument_family"])
# This is all of the instrument family indices, our output to validate on
# Index meaning can be found at https://magenta.tensorflow.org/datasets/nsynth#instrument-families

datatest = rawtest.map(_parseme)  # no synth lead exists in the test data
# This is all the audio frequencies converted to mel, our input to test
dataytest = datatest.map(lambda d: d["instrument_family"])
# This is all of the instrument family indices, our output to predict
# Index meaning can be found at https://magenta.tensorflow.org/datasets/nsynth#instrument-families



def progress_bar(count, total, suffix=''):
    """Prints a progress bar
    Args:
        count  (int): counter of the current step
        total  (int): total number of steps
        suffix (str): to suffix the progress bar
    """
    bar_len = 60
    
    filled_len = int(round(60 * count / float(total)))
    percents = count
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s/%s ...%s\r' % (bar, percents, total, suffix))
    sys.stdout.flush()

def extract_from_f(r, seconds=1):
    """Extracts the mel-spectogram from the input file
    Args:
        filename (str): the name of the input file
        seconds  (int): lenght of audio excerpt in seconds
    Returns:
        features (dict): contains the filename, the mono audio signal, the STFT,
                         the mspec and the labels
    """
    # Read the data
    audio = r["audio"].numpy()
    fs = int(r["sample_rate"])

    audio = audio[np.arange(0, audio.size, 2)]
    audio /= np.max(np.abs(audio))
    audio = np.squeeze(audio)

    # Compute Short Time Fourier Transform
    stft = np.abs(librosa.stft(audio, win_length=1024, hop_length=512,
        center=True))
    # Convert to Mel Spectogram
    mel_spec = librosa.feature.melspectrogram(S=stft, sr=fs/2, n_mels=128)
    # Take the natural logarithm of the Mel Spectogram
    ln_mel_spec = np.log(mel_spec + np.finfo(float).eps)

    seg_dur = 43 * seconds
    spec_list = []
    for idx in range(0, ln_mel_spec.shape[1] - seg_dur + 1, seg_dur):
        spec_list.append(ln_mel_spec[:, idx : (idx + seg_dur)])
    spec_list = np.array(spec_list)
    spec_list = np.squeeze(spec_list)
    mspecs = np.expand_dims(spec_list, axis=2)
    features = {}
    features["audio"] =  r["audio"]
    features["mspec"]    = mspecs
    features["labels"]   = int(r["instrument_family"])

    return features






train_x = []
train_y = []
count = 0
total = 0
print("preprocessing train")
for r in datatrain:
    count+=1
    if int(r["instrument_family"]) != 9:
        total+=1
        feat = extract_from_f(r)
        train_x.append(feat["mspec"])
        train_y.append(feat["labels"])
        progress_bar(count, 289205)

    
    

val_x = []
val_y = []
count = 0
total = 0
print("preprocessing valid")

for r in datavalid:
    count+=1
    if int(r["instrument_family"]) != 9:
        total+=1
        feat = extract_from_f(r)
        val_x.append(feat["mspec"])
        val_y.append(feat["labels"])
        progress_bar(count, 12678)



test_x = []
test_y = []
test_audio= []
count = 0
total = 0
print("preprocessing test")
for r in datatest:
    count+=1
    if int(r["instrument_family"] )!= 9:
        total+=1
        feat = extract_from_f(r)
        test_x.append(feat["mspec"])
        test_y.append(feat["labels"])
        test_audio.append(feat["audio"])
        progress_bar(count, 4096)



train_x = np.array(train_x)
train_y = np.array(train_y)
val_y = np.array(val_y)
val_x = np.array(val_x)


test_y = np.array(test_y)
test_x = np.array(test_x)
test_audio = np.array(test_audio)
#print(train_x.shape,test_x.shape)

np.save("train_x",train_x)
np.save("train_y",train_y)
np.save("val_x",val_x)
np.save("val_y",val_y)

np.save("test_x",test_x)
np.save("test_y",test_y)
np.save("test_audio",test_audio)
