from pylsl import StreamInlet, resolve_stream
import socket 
import numpy as np
from pythonosc import udp_client
import random
upd_ip = "127.0.0.1"
udp_port = 11000
osc_ip = "127.0.0.1"
osc_port = 12000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client = udp_client.SimpleUDPClient(osc_ip, osc_port)

def msg_to_bytes(msg):
    return msg.encode('utf-8')
rt = 5 # samples in package per pull
streams = resolve_stream()
inlet = StreamInlet(streams[0])

DATA = []
#for x in range(10):
    #client.send_message("/filter", random.random())
class Unicorn(object):
    def __init__(self):
        pass
    
    def stream_data(self):
        newl = []
        for i in range(rt):
            sample, timestamp = inlet.pull_sample()   
            newl.append(sample)
            #for i in range(17):
            client.send_message("/filter", sample)
            
            
        return (np.array(newl))
        
while True:
    Unicorn().stream_data()
    #sock.sendto(msg_to_bytes(str(Unicorn().stream_data())), (upd_ip, udp_port))