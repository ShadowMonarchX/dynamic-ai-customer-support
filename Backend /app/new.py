# import requests
# from bs4 import BeautifulSoup

# url = "https://nayanraval.vercel.app/"
# response = requests.get(url)
# soup = BeautifulSoup(response.text, "html.parser")

# text = soup.get_text()

# with open("website_data.txt", "w", encoding="utf-8") as file:
#     file.write(text)

import pandas as pd

df = pd.read_csv('/Users/jenishshekhada/Desktop/Inten/dynamic-ai-customer-support/backend /data/aa_dataset-tickets-multi-lang-5-2-50-version.csv')


print(df.head())

print("\n--- Data Columns ---\n")
print(df.columns)

print("\n--- Data Summary ---")
print(df.describe(include='all'))

print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n--- Entries with Answers ---")
print(df.loc[df['answer'].notna(), ['answer']])

print("\n--- Entries with Subject ---")
print(df.loc[df['subject'].notna(), ['subject']])

print("\n--- Entries with Body ---")
print(df.loc[df['body'].notna(), ['body']])
