# /// script
# requires-python = ">=3.11"
# dependecies = [
#     "requests",
#     "json",
#     "rich",
# ]
# ///

import requests
import json

with open("cookies.txt", "r") as file:
    cookie = file.read().strip()

headers = {
    "cookie": cookie
}

response = requests.get("https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34.json?page=1", headers=headers)

with open("discourse.json", "w") as file:
    json.dump(response.json(), file, indent=4)


#   https://discourse.onlinedegree.iitm.ac.in/search?q=%23courses%3Atds-kb%20after%3A2024-12-31&page=1
#   https://discourse.onlinedegree.iitm.ac.in/search?q=%23courses%3Atds-kb%20after%3A2024-12-31&page=2
