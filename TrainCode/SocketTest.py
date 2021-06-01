import socket

sever = socket.socket()
host = socket.gethostname()
port =12345

sever.bind((host,port))

sever.listen(5)

while (True):
    client,addr = sever.accept()
    print('Link Addr:',addr)
    client.send(b'1')
    client.close()

