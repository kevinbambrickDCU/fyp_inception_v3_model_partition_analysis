import socket
import sys
import numpy as np
import torch
import pickle

from dahuffman import HuffmanCodec
from analysis import server_run, decode_delta, decode, load_huff_dictionary

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port
server_address = ('localhost', 10000)
print(sys.stderr, 'starting up on %s port %s' % server_address)
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)
connect = True
failCount = 0
passCount = 0
PREVIOUS_ARRAY = None
USE_DELTA = True
LAST_EDGE_LAYER = 7
NUM_BINS = 60
RESHAPE_ARRAY_DIMENSIONS = [192,35,35]

codec_path =  'huffman_encoding_config/'+'layer'+str(LAST_EDGE_LAYER)+'/'+'num_bins_'+str(NUM_BINS)
delta_hist = load_huff_dictionary(codec_path+'/delta_hist')
delta_codec = HuffmanCodec.from_frequencies(delta_hist)
frame_one_hist = load_huff_dictionary(codec_path+'/frame_one_hist')
frame_one_codec = HuffmanCodec.from_frequencies(frame_one_hist)

while True:
    # Wait for a connection
    received = ''
    arr = bytearray()
    size = 0

    print(sys.stderr, 'waiting for a connection')
    connection, client_address = sock.accept()

    try:
        print(sys.stderr, 'connection from', client_address)

        # Receive the data in small chunks and retransmit it
        while True:
            data = connection.recv(1024)
            size = size + len(data)
            # received = received + (data.decode('utf-8'))
            arr.extend(data)
            # print(sys.stderr, 'received "%s"' % data)
            if data:
                # print(sys.stderr, 'sending data back to the client')
                connection.sendall(data)
            else:
                print(sys.stderr, 'no more data from', client_address)
                connect = False
                break

    finally:
        # Clean up the connection
        connection.close()
        print('Size of data recieved: ', size)

        # print('Recieved: ', arr)
        if PREVIOUS_ARRAY is not None and USE_DELTA is True:
            #new code
            decoded = delta_codec.decode(arr)
            arr = np.reshape(decoded, [192,35,35])

            decoded_arr = decode(arr, max_num=8, min_num=-8, num_bins=NUM_BINS)
            delta_decoded_arr = decode_delta(PREVIOUS_ARRAY, decoded_arr)
            # print(delta_decoded_arr)
            PREVIOUS_ARRAY = delta_decoded_arr
            result = server_run(torch.Tensor(delta_decoded_arr),LAST_EDGE_LAYER)
        else:
            #new code
            decoded = frame_one_codec.decode(arr)
            arr = np.reshape(decoded, [192,35,35])

            decoded_arr = decode(arr, max_num=8, min_num=-8, num_bins=NUM_BINS)
            PREVIOUS_ARRAY = decoded_arr
            result = server_run(torch.Tensor(decoded_arr),LAST_EDGE_LAYER)

        if result:
            passCount += 1
        else:
            failCount += 1

        print('Total checked: ', passCount + failCount)
        print('Number of correct classifictaions: ', passCount)
        print('Number Failed: ', failCount)
