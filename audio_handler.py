import numpy as np
import pyaudio
import time
import librosa
import wave
import joblib
import pandas as pd

'''
Audio Handler class
args:
    --duration: represents the total duration that the main loop runs for
    --svm: represents the unpickled svm that will identify the crying
    --knn: represents the unpickled knn that will classify the type of crying

    TODO - needs to return some sort of indicator for the event handler
    TODO - the reading of the data needs to be float32 instead of int16.
            the librosa load provides the time series in float32 but with the
            current way of loading live data (self.p.open in int16 for audio quality)
            the mfccs would not be the same unless the feature extraction was loading through the 
            wave module in int16 instead of with librosa.load()
            -> this may be able to be resolved with a higher quality mic
'''
class AudioHandler(object):
    def __init__(self, duration, svm, knn, debug=False, eventmonitor=None):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024 * 2
        self.RECORD_SECONDS = 5
        self.DURATION = duration
        self.frame_to_proc = []
        self.audio_buffer = []
        self.n_mfcc = 10
        self.starttime = time.time()
        self.p = None
        self.stream = None
        self.svm = svm
        self.knn = knn
        self.indetification_stamps = []
        self.mfccs = []
        self.eventmonitor = eventmonitor
    
    def _predict(self, mfccs):
        self.mfccs.append(mfccs)
        prediction = self.svm.predict(mfccs.reshape(1, -1))
        # print('clip mfccs',mfccs)
        # print(mfccs)
        if prediction == 'positive':
            cry_classification = self.knn.predict(mfccs.reshape(1, -1))
            self.indetification_stamps.append((time.time, cry_classification))
            print(f"cry detected -> prediction: {cry_classification}")
            if self.eventmonitor:
                print('cry event')
                self.eventmonitor.set_event('cry_event')
        else: print('no cry detected')

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self, debug=False):
        self.audio_buffer.extend(self.frame_to_proc)
        audio_data = b''.join(self.frame_to_proc)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        mfccs = librosa.feature.mfcc(y=audio_array.astype('float32'), 
                                          sr=self.RATE, n_mfcc=self.n_mfcc,dtype=np.float32).mean(axis=1)
        
        self._predict(mfccs)

        with wave.open('debug_out.wav', 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.audio_buffer))
        mfcc_df = pd.DataFrame(data=self.mfccs, columns=[f'MFCC_{i}' for i in range(1, self.n_mfcc + 1)])
        mfcc_df.to_csv('debug_handler_mfccs.csv', index=False)
        self.audio_buffer = []
        self.stream.close()
        self.p.terminate()
        self.p = None
        self.stream = None

    def callback(self, in_data, frame_count, time_info, flag):
        self.frame_to_proc.append(in_data)
        # print('processing')
        if len(self.frame_to_proc) * self.CHUNK  >= self.RECORD_SECONDS * self.RATE:
            self.audio_buffer.extend(self.frame_to_proc)
            audio_data = b''.join(self.frame_to_proc)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            mfccs = librosa.feature.mfcc(y=audio_array.astype('float32'), 
                                              sr=self.RATE, n_mfcc=self.n_mfcc,dtype=np.float32).mean(axis=1)
            self._predict(mfccs)
            self.frame_to_proc = []
        return None, pyaudio.paContinue

    def mainloop(self):
        while (self.stream.is_active() and (time.time() - self.starttime) < self.DURATION): # if using button you can set self.stream to 0 (self.stream = 0), otherwise you can use a stop condition
            time.sleep(1.0)


# TODO
'''
add event handler
'''
class DemoAudioHandler(object):
    def __init__(self, duration, svm, knn, test_wav, debug=False, eventmonitor=None):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 8000
        self.CHUNK = 1024 * 2
        self.RECORD_SECONDS = 5
        self.DURATION = duration
        self.frame_to_proc = []
        self.audio_buffer = []
        self.n_mfcc = 10
        self.starttime = time.time()
        self.p = None
        self.stream = None
        self.wf = wave.open(test_wav, 'rb')
        self.i = 0
        self.indetification_stamps = []
        self.eventmonitor = eventmonitor
        self.svm = svm
        self.knn = knn

    def _predict(self, mfccs):
        prediction = self.svm.predict(mfccs.reshape(1, -1))
        # print('clip mfccs',mfccs)
        if prediction == 'positive':
            cry_classification = self.knn.predict(mfccs.reshape(1, -1))
            self.indetification_stamps.append((time.time, cry_classification))
            print(f"cry detected -> prediction: {cry_classification}")
            if self.eventmonitor:
                print('cry event')
                self.eventmonitor.set_event('cry_event')
        else: print('no cry detected')

    def start(self):
        self.p = pyaudio.PyAudio()
        self.FORMAT = self.p.get_format_from_width(self.wf.getsampwidth())
        # print(self.FORMAT)
        self.CHANNELS = self.wf.getnchannels()
        self.RATE = self.wf.getframerate()
        # print(self.CHANNELS, self.RATE)
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self, debug=False):
        self.audio_buffer.extend(self.frame_to_proc)
        audio_data = b''.join(self.frame_to_proc)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        mfccs = librosa.feature.mfcc(y=audio_array.astype('float32'), 
                                          sr=self.RATE, n_mfcc=self.n_mfcc,dtype=np.float32).mean(axis=1)
        self._predict(mfccs)
        self.frame_to_proc = []
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        self.frame_to_proc.append(self.wf.readframes(frame_count))
        if len(self.frame_to_proc) * self.CHUNK  >= self.RECORD_SECONDS * self.RATE:
            self.audio_buffer.extend(self.frame_to_proc)
            audio_data = b''.join(self.frame_to_proc)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            mfccs = librosa.feature.mfcc(y=audio_array.astype('float32'), 
                                              sr=self.RATE, n_mfcc=10,dtype=np.float32).mean(axis=1)
            self._predict(mfccs)
            self.frame_to_proc = []
        return None, pyaudio.paContinue

    def mainloop(self):
        while (self.stream.is_active() and (time.time() - self.starttime) < self.DURATION): # if using button you can set self.stream to 0 (self.stream = 0), otherwise you can use a stop condition
            time.sleep(1.0)

if __name__ == "__main__":
    knn = joblib.load('./audio_analysis/live_knn.pkl')
    # svm = joblib.load('./audio_analysis/live_svm.pkl')
    # svm = joblib.load('./audio_analysis/min_max_svm.pkl')
    svm = joblib.load('./audio_analysis/weight_svm.pkl')
    # audio = DemoAudioHandler(30, svm, knn, './audio_analysis/data/test/sounds.wav')
    # audio = DemoAudioHandler(30, svm, knn, 'debug_out.wav')
    audio = AudioHandler(40, svm, knn)
    audio.start()    # open the the stream
    audio.mainloop()  
    audio.stop()