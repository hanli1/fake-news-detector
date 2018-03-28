import pandas as pd
import csv
fakes = pd.read_csv("data/fake.csv")
reals = pd.read_csv("data/real.csv")

num_fakes = fakes.shape[0]
num_reals = reals.shape[0]

combined_list = []

def create_obj(source, title, text):
    return {
        "source": source,
        "title": title,
        "text": text
    }

for index, row in fakes.iterrows():
    source = row['author']
    title = row['title']
    # timestamp = row['published']
    text = row['text']
    obj = create_obj(source, title, text)
    combined_list.append(obj)

for index, row in reals.iterrows():
    if index >= num_fakes:
        break
    source = row['publication']
    # timestamp = row['date']
    title = row['title']
    text = row['content']
    obj = create_obj(source, title, text)
    combined_list.append(obj)

print "combined total: {} samples".format(len(combined_list))
combined = pd.DataFrame(combined_list)
combined.to_csv("data/combined.csv", sep=",")



