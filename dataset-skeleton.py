import os
import pandas as pd

os.chdir('data')
home = os.getcwd()

df = pd.DataFrame(columns=['headline','text','interest_rate','category'])
current_index = 0
for folder in os.listdir():
    category = folder
    os.chdir(os.path.join(home, folder))
    for file in os.listdir():
        with open(file, 'r') as f:
            headline = f.readline()
            text = f.read()
        entry = {'headline': headline, 'text': text, 'interest_rate': pd.NA, 'category': category}
        df = pd.concat([pd.DataFrame([[headline, text, pd.NA, category]], columns=df.columns), df], ignore_index=True)
    os.chdir(home)
df.to_csv('data.csv', index=False)
#for folder in os.listdir():
    