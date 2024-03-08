import pyaudio
import wave
import numpy as np
import librosa
import joblib
from sklearn.metrics import accuracy_score 
import warnings 

#με αυτή την εντολή , αφαιρούμε τυχόν warnings , ώστε να επιστρέφονται μόνο τα predictions 
warnings.filterwarnings("ignore")

#φορτώνουμε το μοντέλο 
model = joblib.load('emotion_classification_model.pkl')

'''δοκιμάζουμε τρία συγκεκριμένα ηχητικά αρχεία , φορτώνοντας ένα κάθε φορά 
CAUTION! το true label μέσα στην audio callback πρέπει να αλλάξει ανάλογα με 
το label κάθε ηχητικού αρχείου .Υπενθυμίζουμε ότι όταν ο 3ος αριθμός είναι 05 = angry 
όταν είναι 02 = calm'''
audio_file_path = 'speech_sentiment_part\\Actor_01\\03-01-05-02-01-01-01.wav'
#audio_file_path = 'speech_sentiment_part\\Actor_08\\03-01-02-01-02-01-08.wav'
#audio_file_path = 'speech_sentiment_part\\Actor_22\\03-01-05-01-01-02-22.wav'
 
'''δημιουργούμε μια συνάρτηση για την εξαγωγή των χαρακτηριστικών που θεωρούμε 
πιο κρίσιμα για το classification (mean_centroid ,std_centroid , mean_mfccs ,
std_mfccs ,  mean_bandwidth , std_bandwidth) '''
def extract_features(audio_file):
    audio, sr = librosa.load(audio_file)

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    mean_centroid = np.mean(centroid)
    std_centroid = np.std(centroid)

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # Extract 13 MFCC coefficients
    mean_mfccs = np.mean(mfccs, axis=1)
    std_mfccs = np.std(mfccs, axis=1)

    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    mean_bandwidth = np.mean(bandwidth)
    std_bandwidth = np.std(bandwidth)

    features = np.array(
        [mean_centroid, std_centroid, mean_mfccs[0], std_mfccs[0], mean_bandwidth, std_bandwidth]
    )
    #μετατρέπουμε το array σε 1 γραμμή , 2 διαστάσεις 
    features = features.reshape(1, -1)

    return features

#ορίζουμε τις παρακάτω μεταβλητές ως global
WINDOW_SIZE = 2048
CHANNELS = 1
RATE = 44100

'''δημιουργούμε μια συνάρτηση , η οποία ανοίγει το audio_file 
με την wave.open σε read mode'''
def load_sound_file_into_memory(path):
    audio_data = wave.open(path, 'rb')
    return audio_data

f = load_sound_file_into_memory(audio_file_path)
#position indicator 
f.setpos(0)

'''δημιουργούμε μια συνάρτηση  , η οποία λαμβάνει τα data από το αρχείο σε
συγκεκριμένο window size , εξάγει τα χαρακτηριστικά όπως έχουν οριστεί από
την extract_features παραπάνω και ύστερα προβλέπει ποιο συναίσθημα εντοπίζει
σε κάθε παράθυρο . Τέλος , δίνοντας του ένα συγκεκριμένο true label (angry/calm) , 
υποδεικνύουμε το σωστό prediction , για να το αξιολογήσουμε μέσω του accuracy '''
def audio_callback(in_data, frame_count, time_info, status):
    block_for_speakers = np.zeros((WINDOW_SIZE, CHANNELS), dtype='int16')

    features = extract_features(audio_file_path)
    features_reshaped = features.reshape(1, -1)
    predicted_emotion = model.predict(features_reshaped)
    
    true_label = "angry"  
    
    accuracy = accuracy_score([true_label], [predicted_emotion])
   
    print("Predicted emotion:", predicted_emotion)
    print('Accuracy:' , accuracy)
    print()

    return (block_for_speakers, pyaudio.paContinue)

#δημιουργούμε το output stream 
p = pyaudio.PyAudio()

#ανοίγουμε το output stream
output = p.open(format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=False,
                frames_per_buffer=WINDOW_SIZE,
                stream_callback=audio_callback)

#ξεκινάμε το output stream 
output.start_stream()

#δημιουργούμε ένα while loop , για να συνεχίζουν τα predictions όσο το output είναι ενεργό
while output.is_active():
    continue
#σταματάμε το output stream 
output.stop_stream()

#κλείνουμε το output stream 
output.close()

#τερματίζουμε το instance του output stream
p.terminate()

