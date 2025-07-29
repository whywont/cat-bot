from flask import Flask, send_from_directory

app = Flask(__name__)
shared_state = {
    "manual_action_id": None
}

@app.route("/move/<dir>")
def move(dir):
    if dir.lower() in ['f', 'b', 'l', 'r', 's']:
        shared_state["manual_action_id"] = dir.upper()
        return f"Queued action: {shared_state['manual_action_id']}"
    return "Invalid"

@app.route("/")
def controller():
    return send_from_directory(".", "controller.html")

def get_manual_action():
    return shared_state["manual_action_id"]

def clear_manual_action():
    shared_state["manual_action_id"] = None

def start_flask():
    app.run(host="0.0.0.0", port=5000)
