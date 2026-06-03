import requests

url = f"https://sheets.googleapis.com/v4/spreadsheets/1mSUVtClmlGmpjo2jp9EFxkkoG_IxS85f/values/test!A1:append"

params = {
    "valueInputOption": "USER_ENTERED",
    "insertDataOption": "INSERT_ROWS",
    "key": "AIzaSyBKE1i0xYorNLhX5HvSENUOF_GRLSamsrI"
}

body = {
    "values": [["hello", "world", "123"]]
}

r = requests.post(url, params=params, json=body)

print(r.status_code)
print(r.text)