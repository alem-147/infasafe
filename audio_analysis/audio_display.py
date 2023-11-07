
"""
    credit: https://github.com/markjay4k
    audio is captured using pyaudio
    then converted from binary data to ints using struct
    then displayed using matplotlib

    scipy.fftpack computes the FFT

    if you don't have pyaudio, then run

    >>> pip install pyaudio

    note: with 2048 samples per chunk, I'm getting 20FPS
    when also running the spectrum, its about 15FPS
"""
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
# from pyqtgraph.Qt import QtGui, QtCore
# import pyqtgraph as pg
import struct
from scipy.fftpack import fft
import sys
import time


class AudioStream(object):
    def __init__(self):

        # stream constants
        self.CHUNK = 1024 * 2
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.pause = False
        # self.p = p
        # stream object
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK,
        )

        # self.stream = self.p.open(format=pyaudio.paFloat32,
        #         channels=self.CHANNELS,
        #         rate=self.RATE,
        #         output=True,
        #         input=True,
        #         stream_callback=self.callback)
        # self.stream.start_stream()
        self.init_plots()
        self.start_plot()

    # def test(self):

    #     print('stream started')
    #     frame_count = 0
    #     start_time = time.time()

    #     while not self.pause:
    #         data = self.stream.read(self.CHUNK)
    #         frame_count += 1
    #         if frame_count%50 == 0:
    #             print(frame_count)
    #     else:
    #         self.fr = frame_count / (time.time() - start_time)
    #         print('average frame rate = {:.0f} FPS'.format(self.fr))
    #         self.exit_app()

    def init_plots(self):

        # x variables for plotting
        x = np.arange(0, 2 * self.CHUNK, 2)
        xf = np.linspace(0, self.RATE, self.CHUNK)

        # create matplotlib figure and axes
        self.fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 7))
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)

        # create a line object with random data
        self.line, = ax1.plot(x, np.random.rand(self.CHUNK), '-', lw=2)

        # create semilogx line for spectrum
        self.line_fft, = ax2.semilogx(
            xf, np.random.rand(self.CHUNK), '-', lw=2)

        # format waveform axes
        ax1.set_title('AUDIO WAVEFORM')
        ax1.set_xlabel('samples')
        ax1.set_ylabel('volume')
        ax1.set_ylim(0, 255)
        ax1.set_xlim(0, 2 * self.CHUNK)
        plt.setp(
            ax1, yticks=[0, 128, 255],
            xticks=[0, self.CHUNK, 2 * self.CHUNK],
        )
        plt.setp(ax2, yticks=[0, 1],)

        # format spectrum axes
        ax2.set_xlim(20, self.RATE / 2)

        # show axes
        # thismanager = plt.get_current_fig_manager()
        # thismanager.window.setGeometry(5, 120, 1910, 1070)
        plt.show(block=False)

    def callback(self, in_data, frame_count, time_info, status):
        global yf, data_np
        audio_data = np.fromstring(in_data, dtype=np.float32)
        data_int = struct.unpack(str(2 * self.CHUNK) + 'B', audio_data)
        data_np = np.array(data_int, dtype='b')[::2] + 128
        yf = fft(data_int)
        return (audio_data, pyaudio.paContinue)


    def start_plot(self):

        print('stream started')
        frame_count = 0
        start_time = time.time()

        while not self.pause:
            data = self.stream.read(self.CHUNK)
            data_int = struct.unpack(str(2 * self.CHUNK) + 'B', data)
            data_np = np.array(data_int, dtype='b')[::2] + 128

            self.line.set_ydata(data_np)

            # compute FFT and update line
            yf = fft(data_int)
            self.line_fft.set_ydata(
                np.abs(yf[0:self.CHUNK]) / (128 * self.CHUNK))

            # update figure canvas
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            frame_count += 1

        else:
            self.fr = frame_count / (time.time() - start_time)
            print('average frame rate = {:.0f} FPS'.format(self.fr))
            self.exit_app()

    def exit_app(self):
        print('stream closed')
        self.p.close(self.stream)

    def onClick(self, event):
        self.pause = True


if __name__ == '__main__':
    # p = pyaudio.PyAudio()
    # ast = AudioStream(p)
    AudioStream()
    # ast.init_plots()
    # ast.start_plot()
    # while ast.stream.is_active():
    #     time.sleep(5)
    #     ast.stream.stop_stream()
    # ast.stream.close()
    # p.terminate()
