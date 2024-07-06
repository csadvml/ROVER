import eventlet
import socketio

sio = socketio.Server()
app = socketio.WSGIApp(sio)

@sio.event
def connect(sid, environ):
    print("connect ", sid)
    send_control((0, 0, 0), '', '')
    sample_data = {}
    sio.emit("get_samples", sample_data, skip_sid=True)

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
