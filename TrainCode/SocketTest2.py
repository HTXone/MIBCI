import socket

s = socket.socket()
host = socket.gethostbyname('192.168.43.28')
port = 44981

s.connect((host,port))
print(s.recv(1024))
s.close()