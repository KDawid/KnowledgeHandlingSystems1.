import os
import pandas as pd
import re
import sys

if(len(sys.argv) != 2):
    print("Please add the csv file name")
    sys.exit(-1)

input_file = sys.argv[1]
directory = "results/"

if not os.path.exists(directory):
    os.makedirs(directory)

regex = re.compile('[^a-zA-Z0-9]')

print("reading %s" % input_file)
ind = 0
for df in pd.read_csv(input_file, sep=',', chunksize=100000):
    df = df.drop(["Unnamed: 0", "domain", "url", "scraped_at",	 "inserted_at", "updated_at", "title", "authors", "keywords", "meta_keywords", "meta_description", "tags", "summary", "source"], axis=1)
    print("chunk %i" % ind)
    for i, value in df['content'].items():
        df['content'][i] = str(df['content'][i]).replace('\n', " ").replace('\r', " ").replace("'", " ").replace('"', ' ').replace(",", " ")
    df.to_csv((directory + "result_%i.csv" % ind), sep=",", index=False)
    ind += 1
