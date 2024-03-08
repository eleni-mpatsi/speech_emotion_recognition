import pyaudio
import numpy as np
import librosa
import joblib
import warnings

warnings.filterwarnings('ignore')

# ορίζουμε τη συσκευή εισόδου ήχου για την εγγραφή δεδομένων ήχου
mic_device_index = 1

WINDOW_SIZE = 1024
CHANNELS = 1
RATE = 44100

# φορτώνουμε το μοντέλο
model = joblib.load('emotion_classification_model.pkl')

#χρησιμοποιούμε την ίδια συνάρτηση με το window_evaluation
def extract_features(audio_data):
    centroid = librosa.feature.spectral_centroid(y=audio_data, sr=RATE)
    mean_centroid = np.mean(centroid)
    std_centroid = np.std(centroid)

    mfccs = librosa.feature.mfcc(y=audio_data, sr=RATE, n_mfcc=13)
    mean_mfccs = np.mean(mfccs, axis=1)
    std_mfccs = np.std(mfccs, axis=1)

    bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=RATE)
    mean_bandwidth = np.mean(bandwidth)
    std_bandwidth = np.std(bandwidth)

    features = np.array([mean_centroid, std_centroid, mean_mfccs[0], std_mfccs[0], mean_bandwidth, std_bandwidth])
    features = features.reshape(1, -1)

    return features

'''σε αυτή την περίπτωση , η callback συνάρτηση λαμβάνει τα in_data από 
την Pyadio , και συγκεκριμένα από το mic_device_index , τα μετατρέπει σε 
type float , εξάγει τα χαρακτηριστικά με την extract_features και κάνει
μία πρόβλεψη για το κάθε παράθυρο'''
def callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.int16)
    audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max  

    features = extract_features(audio_data)
    predicted_emotion = model.predict(features)
    print("Predicted emotion:", predicted_emotion)
    print()

    return (in_data, pyaudio.paContinue)


p = pyaudio.PyAudio()

output = p.open(format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=RATE,
                output=False,
                input=True,
                #σε αυτό το σημείο προσθέτουμε το argument input_device_index
                input_device_index=mic_device_index,
                frames_per_buffer=WINDOW_SIZE,
                stream_callback=callback,
                start=False)

output.start_stream()

while output.is_active() :
    continue

output.stop_stream()
output.close()

p.terminate()

