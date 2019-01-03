import json
from os import listdir
import csv

class FakeNewsCsvResultCreator:
    def __init__(self, configFilePath):
        with open('config.json') as json_data_file:
            config = json.load(json_data_file)
        self.RESULT_FOLDER = config["RESULT_FOLDER"]
        self.CSV_RESULT_FILE_PATH = config["CSV_RESULT_FILE_PATH"]

    def createResultCsv(self):
        result = dict()
        resultFiles = [f for f in listdir(self.RESULT_FOLDER) if f.endswith("_result.json")]
        for file in resultFiles:
            with open(self.RESULT_FOLDER + file) as json_data_file:
                data = json.load(json_data_file)
                for key in data:
                    for i in data[key]:
                        if i not in result:
                            result[i] = dict()
                        if key not in result[i]:
                            result[i][key] = dict()
                        result[i][key][file] = data[key][i]
        for acc_type in result:
            fileName = self.CSV_RESULT_FILE_PATH[:-4] + "_" + acc_type + ".csv"
            head = list(result['test'].keys())
            keyLength = len(head)-1
            with open(fileName, 'wb') as file:
                file.write(",".encode())
                for key in head:
                    if head[keyLength] == key:
                        file.write(key.encode())
                    else:
                        file.write((key + ",").encode())
                file.write("\n".encode())
                methods = list(result[acc_type][next(iter(result[acc_type]))].keys())
                for method in methods:
                    file.write((method[:-5] + ",").encode())
                    for key in head:
                        if head[keyLength] == key:
                            file.write(str(result[acc_type][key][method]).encode())
                        else:
                            file.write((str(result[acc_type][key][method]) + ",").encode())
                    file.write("\n".encode())