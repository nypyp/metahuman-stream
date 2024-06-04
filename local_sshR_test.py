import socket

def server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('0.0.0.0', 8010))
    s.listen(1)
    print("Server is running and listening on port 8010...")
    while True:
        conn, addr = s.accept()
        print(f"Connected by {addr}")
        conn.sendall(b"Hello, this is a message from the local server!")
        conn.close()

if __name__ == "__main__":
    server()
