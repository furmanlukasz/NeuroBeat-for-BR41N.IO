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

from scipy import signal
from scipy.signal import butter, lfilter
n = 2500
rt = 8
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
streams = resolve_stream()
inlet = StreamInlet(streams[0])
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

async def Unicorn():
    newl = []
    stamp = []
    k = rt
    
    for i in range(rt):
        sample, timestamp = inlet.pull_sample()   
        newl.append(sample)
    return np.array(newl)

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
        k = rt
        #y = asyncio.run(Unicorn()).astype(np.float32)
     
        #y = asyncio.run(Unicorn())
        # sample, timestamp = inlet.pull_sample()
        #data = [asyncio.run(Unicorn()) for i in range(10)]
        #data = np.transpose(data)
        #print(y.shape)
        #remap( np.transpose(asyncio.run(Unicorn())), -10000, 10000, -1, 1 )
        
        y[:, :-k] = y[:, k:]
        y[:, -k:] = remap( np.transpose(asyncio.run(Unicorn())), -40, 40, -1, 1 ) 
        #remap( np.transpose(asyncio.run(Unicorn())), -7000, 7000, -1, 1 ) 
        #np.transpose(asyncio.run(Unicorn()))
        
        #print(len(y[0]))
        #print(len(y[0]))
        #filtered_packet = sig_filt.get(packet)
        
        #y2 = lfilter(b, a, data)
        #y2 = np.array([signal.convolve(y[i], filter, mode='same') for i in range(17) ])
        y2 = np.array([lfilter(b, a, y[i]) for i in range(17)])
        self.program['a_position'].set_data(y2.ravel().astype(np.float32))
        self.update()

    def on_draw(self, event):
        gloo.clear()
        self.program.draw('line_strip')
        

if __name__ == '__main__':
    c = Canvas()
    app.run()
    #asyncio.run(Unicorn())

