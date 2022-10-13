import requests


def test_health(url):
    response = requests.get(f"{url}/health_check")

    assert response.status_code == 200
    assert response.json()["message"] == "OK"


def test_api(files, url):
    files_opened = []
    for file in files:
        files_opened.append(("data", open(file, "rb")))
    response = requests.post(
        url=f"{url}/predict",
        files=files_opened,
    )
    print(response.json())

    assert response.status_code == 200
    assert response.json()["message"] == "Successful"
