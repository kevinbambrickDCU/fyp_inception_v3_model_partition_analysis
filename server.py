import socket
import sys
import pickle

from analysis import server_run

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port
server_address = ('localhost', 10000)
print(sys.stderr, 'starting up on %s port %s' % server_address)
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

connect = True

received = ''

arr = bytearray()

size = 0

while connect:
    # Wait for a connection
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
            print(sys.stderr, 'received "%s"' % data)
            if data:
                print(sys.stderr, 'sending data back to the client')
                connection.sendall(data)
            else:
                print(sys.stderr, 'no more data from', client_address)
                connect = False
                break
            
    finally:
        # Clean up the connection
        connection.close()    

print('Size of data recieved: ', size)

#print('Recieved: ', arr)

arr = pickle.loads(arr)
#print(arr)


server_run(arr)