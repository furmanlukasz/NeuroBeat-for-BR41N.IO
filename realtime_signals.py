#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vispy: gallery 2
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

"""
Multiple real-time digital signals with GLSL-based clipping.
"""
from pylsl import StreamInlet, resolve_stream
import time
from vispy import gloo
from vispy import app
import numpy as np
import math
import asyncio
import scipy
import threading
import _thread

import socket 

upd_ip = "127.0.0.1"
udp_port = 7000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

from scipy import signal
from scipy.signal import butter, lfilter
from matplotlib import pyplot as plt

n = 2500 # displayed sample in gui window
rt = 8 # samples in package per pull
streams = resolve_stream()
inlet = StreamInlet(streams[0])

DATA = []

class Unicorn(object):
    def __init__(self):
        pass
    
    def stream_data(self):
        newl = []
        for i in range(rt):
            sample, timestamp = inlet.pull_sample()   
            newl.append(sample)
        #await asyncio.sleep(1)
        return np.transpose(np.array(newl))

    def buffer(self):
        while True:
            sample, timestamp = inlet.pull_sample()   
            DATA.append(sample)
        
        
        
            

    def get_data(self,lenght):
        data = []    
        for i in range(lenght):
            sample, timestamp = inlet.pull_sample()   
            data.append(sample)
        
        return np.transpose(np.array(data))

    def show_data(self,data):
        
        plt.title("Unicorn package Data") 
        plt.xlabel("x axis caption") 
        plt.ylabel("y axis caption") 
        #print(len(data[0]))
        for j in range(0,8):
            #print(data[j])
            y = data[j]
            x = np.arange(0,len(data[j]))
            #plt.plot(x,y) 
        #plt.show()
        #await asyncio.sleep(1)
    
    async def main(self,data):
        await asyncio.gather(self.stream_data(),self.show_data(data))

      



class Filter:
  """
    This class creates filters. The input in creating object of the class is:
    order - order of the filter - int
    crit_freq - critical frequencies - array of length 2
    btype - type of the filter, default is "bandpass" - string with possible values of "bandpass", "lowpass", "highpass", "bandstop"
    fs - sampling frequency, default is 250 - int

    n_filters counts the number of created objects of the class
    n_channels is an integer of number of channels which should be filtered
 """

n_filters = 0
n_channels = 17

def __init__(self, order, crit_freq, btype='bandpass', fs=250):
    self.order = order
    self.crit_freq = crit_freq
    self.btype = btype
    self.fs = fs
    self.b, self.a = signal.iirfilter(self.order, self.crit_freq, rp=None, rs=None, btype=self.btype, analog=False, ftype='butter', output='ba', fs=self.fs)

    Filter.n_filters += 1

#@classmethod
def set_channels(cls, n_channels):
    #method to set number of channels for all filters
    cls.n_channels = n_channels

def print_parameters(self):
    #method to print parameters of a filter
    print("Order of filter: {} \n".format(self.order))
    print("Critical frequency: {} \n".format(self.crit_freq))
    print("Type: {} \n".format(self.btype))
    print("Sampling frequency: {} \n".format(self.fs))

def set_min_threshold(self, min_threshold):
    #method to manualy set minimum value of threshold used in rescaling (in calibration process)
    self.min_threshold = min_threshold

def set_max_threshold(self, max_threshold):
    #method to manualy set maximum value of threshold used in rescaling (in calibration process)
    self.max_threshold = max_threshold

def get_data(self, wave):
    """
        method used to use filter in preprocessing, input is wave to filter
        minimum and maximum value of threshold is automatically set in preprocessing to be further utilize in calibration process
        returns filtered data in the form of numpy array
    """
    data = np.array([lfilter(self.b, self.a, wave[i]) for i in range(self.n_channels)])
    self.min_threshold = np.min(data)
    self.max_threshold = np.max(data)
    return data

def get_data_rescaled(self, wave):
    """
        used to filter data in training. Rescales the input data according to previously set thresholds and return filtered numpy array
    """
    m = (self.max_threshold - self.min_threshold)/(np.max(wave) - np.min(wave))
    b = self.min_threshold - m * np.min(wave)
    wave = m * wave + b
    return np.array([lfilter(self.b, self.a, wave[i]) for i in range(self.n_channels)])



#filter = signal.firwin(200, [0.1, 0.9], pass_zero=False)
filter = signal.firwin(400, [0.01, 0.06], pass_zero=False)
b, a =  scipy.signal.iirfilter(5, [7, 8.5], rp=None, rs=None, btype='bandpass', analog=False, ftype='butter', output='ba', fs=250) # bandpas 5th order 2-15 Hz


def remap( x, oMin, oMax, nMin, nMax ):

    #range check
    if oMin == oMax:
        print("Warning: Zero input range")
        return None

    if nMin == nMax:
        print("Warning: Zero output range")
        return None

    #check reversed input range
    reverseInput = False
    oldMin = min( oMin, oMax )
    oldMax = max( oMin, oMax )
    if not oldMin == oMin:
        reverseInput = True

    #check reversed output range
    reverseOutput = False   
    newMin = min( nMin, nMax )
    newMax = max( nMin, nMax )
    if not newMin == nMin :
        reverseOutput = True

    portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
    if reverseInput:
        portion = (oldMax-x)*(newMax-newMin)/(oldMax-oldMin)

    result = portion + newMin
    if reverseOutput:
        result = newMax - portion

    return result


# Number of cols and rows in the table.
nrows = 17
ncols = 1

# Number of signals.
m = nrows*ncols

# Number of samples per signal.

# Various signal amplitudes.
amplitudes = .1 + .2 * np.random.rand(m, 1).astype(np.float32)

# Generate the signals as a (m, n) array.

y = amplitudes * np.random.randn(m, n).astype(np.float32)

# Color of each vertex (TODO: make it more efficient by using a GLSL-based
# color map and the index).
color = np.repeat(np.random.uniform(size=(m, 3), low=.5, high=.9),
                  n, axis=0).astype(np.float32)

# Signal 2D index of each vertex (row and col) and x-index (sample index
# within each signal).
index = np.c_[np.repeat(np.repeat(np.arange(ncols), nrows), n),
              np.repeat(np.tile(np.arange(nrows), ncols), n),
              np.tile(np.arange(n), m)].astype(np.float32)

VERT_SHADER = """
#version 120

// y coordinate of the position.
attribute float a_position;

// row, col, and time index.
attribute vec3 a_index;
varying vec3 v_index;

// 2D scaling factor (zooming).
uniform vec2 u_scale;

// Size of the table.
uniform vec2 u_size;

// Number of samples per signal.
uniform float u_n;

// Color.
attribute vec3 a_color;
varying vec4 v_color;

// Varying variables used for clipping in the fragment shader.
varying vec2 v_position;
varying vec4 v_ab;

void main() {
    float nrows = u_size.x;
    float ncols = u_size.y;

    // Compute the x coordinate from the time index.
    float x = -1 + 2*a_index.z / (u_n-1);
    vec2 position = vec2(x - (1 - 1 / u_scale.x), a_position)+vec2(0.0,-0.5);

    // Find the affine transformation for the subplots.
    vec2 a = vec2(1./ncols, 1./nrows)*.9;
    vec2 b = vec2(-1 + 2*(a_index.x+.5) / ncols,
                  -1 + 2*(a_index.y+.5) / nrows);
    // Apply the static subplot transformation + scaling.
    gl_Position = vec4(a*u_scale*position+b, 0.0, 1.0);

    v_color = vec4(a_color, 1.);
    v_index = a_index;

    // For clipping test in the fragment shader.
    v_position = gl_Position.xy;
    v_ab = vec4(a, b);
}
"""

FRAG_SHADER = """
#version 120

varying vec4 v_color;
varying vec3 v_index;

varying vec2 v_position;
varying vec4 v_ab;

void main() {
    gl_FragColor = v_color;

    // Discard the fragments between the signals (emulate glMultiDrawArrays).
    if ((fract(v_index.x) > 0.) || (fract(v_index.y) > 0.))
        discard;

    // Clipping test.
    vec2 test = abs((v_position.xy-v_ab.zw)/v_ab.xy);
    //if ((test.x > 1) || (test.y > 1))
     //   discard;
}
"""



class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, title='Use your wheel to zoom!',
                            keys='interactive')
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['a_position'] = y.reshape(-1, 1)
        self.program['a_color'] = color
        self.program['a_index'] = index
        self.program['u_scale'] = (1., 1.)
        self.program['u_size'] = (nrows, ncols)
        self.program['u_n'] = n

        gloo.set_viewport(0, 0, *self.physical_size)

        self._timer = app.Timer('auto', connect=self.on_timer, start=True) # connect=self.on_timer

        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

        self.show()



    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)

    def on_mouse_wheel(self, event):
        dx = np.sign(event.delta[1]) * .05
        scale_x, scale_y = self.program['u_scale']
        scale_x_new, scale_y_new = (scale_x * math.exp(2.5*dx),
                                    scale_y * math.exp(0.0*dx))
        self.program['u_scale'] = (max(1, scale_x_new), max(1, scale_y_new))
        self.update()

    def on_timer(self, event):
        """Add some data at the end of each signal (real-time signals)."""
        
        o = Unicorn()
        data = o.get_data(rt)
        k = len(data[0])
        y[:, :-k] = y[:, k:]
        y[:, -k:] = remap((data), -40, 40, -1, 1 ) 
        t2 = _thread.start_new_thread(printT, ())
        #y2 = np.array([lfilter(b, a, y[i]) for i in range(17)])
        self.program['a_position'].set_data(y.ravel().astype(np.float32))
        self.update()

    def on_draw(self, event):
        gloo.clear()
        self.program.draw('line_strip')
        

def printT():
    print(DATA)

if __name__ == '__main__':
    c = Canvas()
    o = Unicorn()
    data = o.get_data(250)
    t1 = _thread.start_new_thread(Unicorn().buffer, (data,))
   
    app.run()
    
    #asyncio.run(Unicorn())

