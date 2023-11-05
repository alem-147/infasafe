import numpy as np
import pyaudio
import time
import librosa
import wave

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
    def __init__(self, duration, svm, knn, debug=False):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
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
        self.svm = svm
        self.knn = knn

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
        prediction = self.svm.predict(mfccs.reshape(1, -1))
        if prediction == 'positive':
            cry_classification = self.knn.predict(mfccs.reshape(1, -1))
            print('cry guess', cry_classification)
        # print('prediction',prediction,'end mfccs',mfccs)
        self.frame_to_proc = []

        with wave.open('debug_out.wav', 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.audio_buffer))
        self.stream.close()
        self.p.terminate()

    #TODO - increase the buffer to 5 sec
    def callback(self, in_data, frame_count, time_info, flag):
        self.frame_to_proc.append(in_data)
        # print('processing')
        if len(self.frame_to_proc) * self.CHUNK  >= self.RECORD_SECONDS * self.RATE:
            self.audio_buffer.extend(self.frame_to_proc)
            audio_data = b''.join(self.frame_to_proc)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            mfccs = librosa.feature.mfcc(y=audio_array.astype('float32'), 
                                              sr=self.RATE, n_mfcc=self.n_mfcc,dtype=np.float32).mean(axis=1)
            prediction = self.svm.predict(mfccs.reshape(1, -1))
            # print('prediction', prediction,'clip mfccs',mfccs)
            if prediction == 'positive':
                cry_classification = self.knn.predict(mfccs.reshape(1, -1))
                print('cry guess', cry_classification)
            self.frame_to_proc = []
        return None, pyaudio.paContinue

    def mainloop(self):
        while (self.stream.is_active() and (time.time() - self.starttime) < self.DURATION): # if using button you can set self.stream to 0 (self.stream = 0), otherwise you can use a stop condition
            time.sleep(.1)



#for now this works but will need to add some filtering to get out the noise
class DemoAudioHandler(object):
    def __init__(self, duration, svm, knn, test_wav, debug=False):
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
        self.mfccs = None
        self.p = None
        self.stream = None
        self.wf = wave.open(test_wav, 'rb')
        self.i = 0

    def start(self):
        self.p = pyaudio.PyAudio()
        self.FORMAT = self.p.get_format_from_width(self.wf.getsampwidth())
        print(self.FORMAT)
        self.CHANNELS = self.wf.getnchannels()
        self.RATE = self.wf.getframerate()
        print(self.CHANNELS, self.RATE)
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
        self.mfccs = librosa.feature.mfcc(y=audio_array.astype('float32'), 
                                          sr=self.RATE, n_mfcc=self.n_mfcc,dtype=np.float32).mean(axis=1)
        print('end mfccs',self.mfccs)
        self.frame_to_proc = []

        with wave.open('debug_out.wav', 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.audio_buffer))
        self.stream.close()
        self.p.terminate()

    #TODO - increase the buffer to 5 sec
    def callback(self, in_data, frame_count, time_info, flag):
        # self.frame_to_proc.append(in_data)
        self.frame_to_proc.append(self.wf.readframes(frame_count))
        # print('processing')
        if len(self.frame_to_proc) * self.CHUNK  >= self.RECORD_SECONDS * self.RATE:
            # print(self.frame_to_proc)
            self.audio_buffer.extend(self.frame_to_proc)
            audio_data = b''.join(self.frame_to_proc)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            self.mfccs = librosa.feature.mfcc(y=audio_array.astype('float32'), 
                                              sr=8000, n_mfcc=10,dtype=np.float32).mean(axis=1)
            prediction = svm.predict(self.mfccs.reshape(1, -1))
            with wave.open(f"streamfile_{self.i}.wav", 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(audio_data)
            self.i += 1
            print('prediction',prediction,'clip mfccs',self.mfccs)
            if prediction == 'positive':
                cry_classification = knn.predict(self.mfccs.reshape(1, -1))
                print('cry guess', cry_classification)
            self.frame_to_proc = []
        return None, pyaudio.paContinue

    def mainloop(self):
        while (self.stream.is_active() and (time.time() - self.starttime) < self.DURATION): # if using button you can set self.stream to 0 (self.stream = 0), otherwise you can use a stop condition
            time.sleep(.1)