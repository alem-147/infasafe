import numpy as np
import pyaudio
import time
import librosa
import wave
import pickle

#for now this works but will need to add some filtering to get out the noise
class AudioHandler(object):
    def __init__(self, duration, svm, debug=False):
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
        self.prediction = svm.predict(mfccs.reshape(1, -1))
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
            self.prediction = svm.predict(mfccs.reshape(1, -1))
            # print('prediction', prediction,'clip mfccs',mfccs)
            self.frame_to_proc = []
        return None, pyaudio.paContinue

    def mainloop(self):
        while (self.stream.is_active() and (time.time() - self.starttime) < self.DURATION): # if using button you can set self.stream to 0 (self.stream = 0), otherwise you can use a stop condition
            time.sleep(1.0)
