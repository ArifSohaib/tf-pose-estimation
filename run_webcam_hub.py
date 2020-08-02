import logging 
import sys
from signalrcore.hub_connection_builder import HubConnectionBuilder

def RecieveConnId(message):
    logging.info(message)
def RecieveMessage(username, message):
    logging.info(f"username:{username}, message:{message}")

def input_with_default(input_text, default_value):
    value = input(input_text.format(default_value))
    return default_value if value is None or value.strip() == "" else value


server_url = input_with_default("Enter your server url(default: {0}): ","http://10.0.0.45:5000/chatHub")
username = input_with_default("Enter your username (default: {0}): ", "arifsohaib")

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

hub_connection = HubConnectionBuilder().with_url(server_url, options={"verify_ssl":False})\
    .configure_logging(logging.DEBUG, socket_trace=True, handler=handler)\
    .with_automatic_reconnect({"type":"interval", "keep_alive_interval":10, "intervals":[1,3,5,7]}).build()

hub_connection.on_open(lambda:print("connection opened and handshake recieved readt to send messages"))
hub_connection.on_close(lambda:print("connection closed"))

hub_connection.on("RecieveMessage", RecieveMessage)
hub_connection.on("RecieveConnId", RecieveConnId)

hub_connection.start()

message = None

while message != "exit()":
    message = input(">>")
    if message is not None and message != "" and message != "exit()":
        hub_connection.send("SendMessage", [username, message])
hub_connection.stop()
sys.exit()

