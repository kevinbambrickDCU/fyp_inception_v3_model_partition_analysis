import socket
import sys
import numpy as np

from analysis import server_run, decode_delta

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
previous_array = None
use_delta = True

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
            #print(sys.stderr, 'received "%s"' % data)
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

        arr = np.frombuffer(arr, dtype=np.int8)
        arr = np.reshape(arr, [192, 35, 35])
        if previous_array is not None and use_delta is True:
            arr = decode_delta(previous_array, arr)

        print(arr)
        previous_array = arr
        # print('previous array: ', previous_array)
        result = server_run(arr)
        if(result):
            passCount += 1
        else:
            failCount += 1

        print('Total checked: ', passCount+failCount)
        print('Number of correct classifictaions: ', passCount)
        print('Number Failed: ', failCount)

