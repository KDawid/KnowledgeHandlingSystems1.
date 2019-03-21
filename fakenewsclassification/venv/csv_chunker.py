import os
import pandas as pd
import re
import sys

if(len(sys.argv) != 3):
    print("Please add the csv file name")
    sys.exit(-1)

input_file = sys.argv[1]
directory = sys.argv[2] + "/"

if not os.path.exists(directory):
    os.makedirs(directory)

if not os.path.exists(directory + "contents/"):
    os.makedirs(directory + "contents/")

if not os.path.exists(directory + "types/"):
    os.makedirs(directory + "types/")

regex = re.compile('[^a-zA-Z0-9]')

print("reading %s" % input_file)
ind = 0
for df in pd.read_csv(input_file, sep=',', chunksize=100000):
    df = df.drop(["Unnamed: 0", "domain", "url", "scraped_at",	 "inserted_at", "updated_at", "title", "authors", "keywords", "meta_keywords", "meta_description", "tags", "summary", "source"], axis=1)
    contents = df.drop(["type"], axis=1)
    types = df.drop(["content"], axis=1)
    print("chunk %i" % ind)
    for i, value in contents['content'].items():
        contents['content'][i] = str(contents['content'][i]).replace('\n', " ").replace('\r', " ").replace("'", " ").replace('"', ' ').replace(",", " ")
    contents.to_csv((directory + "contents/content_%i.csv" % ind), sep=",", index=False)
    types.to_csv((directory + "types/type_%i.csv" % ind), sep=",", index=False)
    ind += 1
