import requests
from bs4 import BeautifulSoup

url = "https://nayanraval.vercel.app/"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

text = soup.get_text()

with open("website_data.txt", "w", encoding="utf-8") as file:
    file.write(text)
