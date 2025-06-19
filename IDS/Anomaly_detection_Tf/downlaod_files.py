import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin

BASE_URL = "http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/Dataset/CSV/CSV/"
LOCAL_ROOT = "/Users/kushalprakash/Downloads/Dataset"

def download_dir(url, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    resp = requests.get(url); resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a['href']
        if href.endswith('.csv'):  # only CSV links
            # download…
            if href in ("../", "/"): continue
            full_url = urljoin(url, href)
            if href.endswith("/"):
                download_dir(full_url, os.path.join(local_dir, href.rstrip("/")))
            elif href.lower().endswith(".csv"):
                local_path = os.path.join(local_dir, href)
                if not os.path.exists(local_path):
                    print(f"→ {full_url}")
                    r = requests.get(full_url); r.raise_for_status()
                    with open(local_path, "wb") as f:
                        f.write(r.content)

if __name__ == "__main__":
    download_dir(BASE_URL, LOCAL_ROOT)
    print("Done!")
