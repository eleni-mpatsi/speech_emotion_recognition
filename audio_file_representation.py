import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle

class AudioFileRepresentation:    
    '''μέσα από την κλάση εξάγουμε κάποια χαρακτηριστικά που θα είναι
       χρήσιμα για την εκπαίδευση του μοντέλου '''    
    def __init__(self, file_path, sr=44100, n_fft=2048, hop_length=1024, keep_audio=False, keep_aux=False):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        s, sr = librosa.load( file_path)
        self.audio = s
        self.extract_power_spectrum()
        self.make_useful_audio_mask()
        self.make_useful_spectrum()
        self.make_useful_area_features()
        self.name = file_path.split( '\\' )[-1]
        if not keep_audio:
            del self.audio
        if not keep_aux:
            del self.power_spectrum
            del self.useful_spectrum
            del self.spectral_magnitude
            del self.useful_bandwidth
            del self.useful_centroid
            del self.useful_mask
    
    def extract_power_spectrum(self):
        p = librosa.stft(self.audio, n_fft=2048, hop_length=1024)
        self.spectral_magnitude, _ = librosa.magphase(p)
        self.power_spectrum = librosa.amplitude_to_db( np.abs(p), ref=np.max )
    
    def make_useful_audio_mask(self):
        self.rms = librosa.feature.rms(S=self.spectral_magnitude)
        rms = self.rms[0]
        self.useful_mask = np.zeros( rms.size )
        self.useful_mask[ rms > 0.005 ] = 1
        self.useful_mask = self.useful_mask.astype(int)
    
    def make_useful_spectrum(self):
        self.useful_spectrum = self.power_spectrum[:,self.useful_mask == 1]
    
    def plot_spectrum(self, range_low=20, range_high=5000):
        fig , plt_alias =  plt.subplots()
        librosa.display.specshow(self.power_spectrum, sr=self.sr, x_axis='time', y_axis='linear', ax=plt_alias)
        plt_alias.set_ylim([range_low, range_high])
    
    def plot_save_spectrum(self, figure_file_name='test.png', range_low=20, range_high=5000):
        fig , plt_alias =  plt.subplots()
        librosa.display.specshow(self.power_spectrum, sr=self.sr, x_axis='time', y_axis='linear', ax=plt_alias)
        plt_alias.set_ylim([range_low, range_high])
        plt.savefig( figure_file_name , dpi=300 )
    
    
    def plot_useful_spectrum(self, range_low=20, range_high=5000):
        fig , plt_alias =  plt.subplots()
        librosa.display.specshow(self.power_useful_spectrum, sr=self.sr, x_axis='time', y_axis='linear', ax=plt_alias)
        plt_alias.set_ylim([range_low, range_high])
    
    def plot_save_useful_spectrum(self, figure_file_name='test.png', range_low=20, range_high=5000):
        fig , plt_alias =  plt.subplots()
        librosa.display.specshow(self.power_useful_spectrum, sr=self.sr, x_axis='time', y_axis='linear', ax=plt_alias)
        plt_alias.set_ylim([range_low, range_high])
        plt.savefig( figure_file_name , dpi=300 )
    
    def make_useful_area_features(self):
        tmp_features = []
        c = librosa.feature.spectral_centroid(y=self.audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        self.useful_centroid = c[0][ self.useful_mask == 1 ]
        self.mean_centroid = np.mean( self.useful_centroid )
        self.std_centroid = np.std( self.useful_centroid )
        tmp_features.append(self.mean_centroid)
        tmp_features.append(self.std_centroid)
        
        b = librosa.feature.spectral_bandwidth(y=self.audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        self.useful_bandwidth = b[0][ self.useful_mask == 1 ]
        self.mean_bandwidth = np.mean( self.useful_bandwidth )
        self.std_bandwidth = np.std( self.useful_bandwidth )
        tmp_features.append(self.mean_bandwidth)
        tmp_features.append(self.std_bandwidth)
        
        c = librosa.feature.spectral_contrast(y=self.audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        self.useful_contrast = c[0][ self.useful_mask == 1 ]
        self.mean_contrast = np.mean( self.useful_contrast )
        self.std_contrast = np.std( self.useful_contrast )
        tmp_features.append(self.mean_contrast)
        tmp_features.append(self.std_contrast)
        
        f = librosa.feature.spectral_flatness(y=self.audio, n_fft=self.n_fft, hop_length=self.hop_length)
        self.useful_flatness = f[0][ self.useful_mask == 1 ]
        self.mean_flatness = np.mean( self.useful_flatness )
        self.std_flatness = np.std( self.useful_flatness )
        tmp_features.append(self.mean_flatness)
        tmp_features.append(self.std_flatness)
        
        f = librosa.feature.spectral_rolloff(y=self.audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        self.useful_rolloff = f[0][ self.useful_mask == 1 ]
        self.mean_rolloff = np.mean( self.useful_rolloff )
        self.std_rolloff = np.std( self.useful_rolloff )
        tmp_features.append(self.mean_rolloff)
        tmp_features.append(self.std_rolloff)
        
        m = librosa.feature.mfcc( y=self.audio , sr=self.sr )
        self.useful_mfcc = m
        self.usefull_mfcc_normalised = (m-np.min(m))/(np.max(m)-np.min(m))
        self.useful_mfcc_profile = np.mean( self.usefull_mfcc_normalised , axis=1 )
        tmp_features.extend( list(self.useful_mfcc_profile) )
        self.features = np.reshape( tmp_features, (1, len(tmp_features)) )

#%%

# ορίζουμε το filepath από το οποίο θα πάρουμε τα ηχητικά αρχεία 
main_folder = 'speech_sentiment_part'

actor_folders = os.listdir( main_folder )

audio_samples = []
'''ορίζουμε ένα for loop , το οποίο διατέχει όλα τα αρχεία ήχου μέσα στο
actor_folders και κάνει append τα χαρακτηριστικά μέσα στο audio samples '''
for a in actor_folders:
    if a[:5] == 'Actor':
        print('running for actor: ' + a)
        for f in os.listdir( os.path.join( main_folder , a ) ):
            print('running for file: ' + f)
            audio_samples.append( AudioFileRepresentation( os.path.join( main_folder , a , f ) ) )

print('Number of audio samples:', len(audio_samples))

#αποθηκεύουμε τα χαρακτηριστικά σε μορφή pickle
with open('audio_representations.pickle', 'wb') as handle:
    pickle.dump(audio_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

# επιβεβαιώνουμε το μέγεθος 
print( 'size of saved structure: ' + str(len(pickle.dumps(audio_samples, -1))) )

#%%
with open('audio_representations.pickle', 'rb') as handle:
    loaded_structure = pickle.load(handle)

'''
Filename identifiers
    Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    Vocal channel (01 = speech, 02 = song).
    Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
    Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
    Repetition (01 = 1st repetition, 02 = 2nd repetition).
    Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
'''
# θέτουμε επιπλέον σημασιολογικές πληροφορίες 
for r in loaded_structure:
    print( 20*'-' + '\n' + r.name )
    useful_name = r.name.split('.')[0]
    splt = useful_name.split('-')
    r.actor_ID = splt[-1]
    r.female = int(r.actor_ID)%2 == 0
    r.modality = splt[0]
    r.vocal_channel = splt[1]
    r.emotion = splt[2]
    r.emotional_intensity = splt[3]
    r.statement = splt[4]
    r.repetition = splt[5]
    r.phrase_ID = splt[4]
    print( 'actor id: ' + r.actor_ID )
    print( 'female: ' + str(r.female) )
    print ('modality' , r.modality)
    print ('vocal_channel' , r.vocal_channel)
    print ('emotion' , r.emotion)


# αποθηκεύουμε το ανανεωμένο αρχείο σε μορφή pickle
with open('sem_data_representations.pickle', 'wb') as handle:
    pickle.dump(loaded_structure, handle, protocol=pickle.HIGHEST_PROTOCOL)

print( 'size of saved structure: ' + str(len(pickle.dumps(loaded_structure, -1))) )

#%%
#φορτώνουμε το ανανεωμένο αρχείο με τις σημασιολογικές πληροφορίες
with open('sem_data_representations.pickle', 'rb') as handle:
    loaded_structure = pickle.load(handle)

# αρχικοποιούμε τα πιο χρήσιμα χαρακτηριστικά 
tmp_dict = {
    'actor_ID': [],
    'female': [],
    'mean_centroid': [],
    'std_centroid': [],
    'mean_mfccs': [],
    'std_mfccs': [],
    'mean_bandwidth': [],
    'std_bandwidth': [],
    'emotion': [],
    'features': []
}

#κάνουμε append τα χαρακτηριστικά μέσα στις λίστες που υπάρχουν στο tmp_dict
for r in loaded_structure:
    tmp_dict['actor_ID'].append( r.actor_ID )
    tmp_dict['female'].append( r.female )
    tmp_dict['mean_centroid'].append( r.mean_centroid )
    tmp_dict['mean_mfccs'].append( r.std_centroid )
    tmp_dict['std_mfccs'].append( r.std_centroid )
    tmp_dict['std_centroid'].append( r.std_centroid )
    tmp_dict['mean_bandwidth'].append( r.mean_bandwidth )
    tmp_dict['std_bandwidth'].append( r.std_bandwidth )
    tmp_dict['emotion'].append( r.emotion )
    tmp_dict['features'].append( r.features )
    
    
'''δημιουργούμε ένα dataframe , το οποίο έχει ως στήλες τα χαρακτηριστικά 
που περιέχονται στο tmp_dict και αφαιρούμε τις NaN τιμές '''
df = pd.DataFrame( tmp_dict ).dropna()

# αποθηκεύουμε το dataframe σε ένα pickle αρχείο 
df.to_pickle('prepared_dataframe.pickle')