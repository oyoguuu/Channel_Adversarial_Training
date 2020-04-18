import librosa

y, sr = librosa.load('i1.wav',sr=None)
mfccs = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=24)
