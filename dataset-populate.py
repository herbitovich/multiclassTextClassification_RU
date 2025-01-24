import pandas as pd
from sklearn.utils import shuffle

data = pd.read_csv('news.csv')
data = shuffle(data)
data.reset_index(inplace=True, drop=True)

for ind, row in data.iterrows():
    if not pd.isnull(row['interest_rate']): continue
    print(row['headline'].strip())
    ans = input("How interesting does the headline above sound? (1-5): ")
    if ans == "exit":
        break    
    elif ans.isdigit():
        ir = int(ans)
        if 1 <= ir <= 5:
            data.at[ind, 'interest_rate'] = ir
        else:
            print("Wrong format.")
    print()


data.to_csv('news.csv', index=False)