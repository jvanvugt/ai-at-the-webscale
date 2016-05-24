import requests
import json

HOSTNAME = "spider-ai.ddns.net"
PORT = "80"
SERVER = "http://" + HOSTNAME + ":" + PORT

TEAM_PASSWORD = ""
TEAM_NAME = ""

def authenticate(name, password):
    global TEAM_PASSWORD
    TEAM_PASSWORD = password
    global TEAM_NAME
    TEAM_NAME = name

def validate_credentials(*args):
    if TEAM_PASSWORD == "" or TEAM_NAME == "":
        raise Exception('Team name or password is not set, please call api.authenticate first.')

def validate_ids(run_id, request_number):
    if run_id < 0:
        raise ValueError("run_id has an invalid value of {}".format(run_id))

    if request_number < 0 or request_number > 9999:
        raise ValueError("request_number has an invalid value of {}".format(request_number))

def get_context(run_id, request_number):
    validate_credentials()
    validate_ids(run_id,request_number)

    params = {
        "team_id": TEAM_NAME,
        "team_password": TEAM_PASSWORD,
        "run_id": run_id,
        "request_number": request_number
    }

    r = requests.get(SERVER + "/get_context", params=params)
    if not r.status_code == 200:
        print r.text
        raise Exception("Something went wrong, see message above.")
    else:
        return r.json()

def serve_page(run_id, request_number, header, language, adtype, color, price):
    validate_credentials()
    validate_ids(run_id, request_number)

    data = {
        "team_id": TEAM_NAME,
        "team_password": TEAM_PASSWORD,
        "run_id": run_id,
        "request_number": request_number,
        "header": header,
        "language": language,
        "adtype": adtype,
        "color": color,
        "price": price
    }

    r = requests.post(SERVER + "/serve_page", data=data)
    if not r.status_code == 200:
        print r.text
        raise Exception("Something went wrong, see message above.")
    else:
        return r.json()

def reset_leaderboard():
    data = {
        "team_id": TEAM_NAME,
        "team_password": TEAM_PASSWORD
    }
    r = requests.post(SERVER + "/reset_leaderboard", data=data)
    if not r.status_code == 200:
        print r.text
        raise Exception("Something went wrong, see message above.")
    else:
        return r.json()
