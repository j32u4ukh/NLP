import os
import re
from datetime import datetime

import numpy as np


class NewsVector:
    def __init__(self):
        self.root = "Crescent/data/"
        self.news_path = os.path.join(self.root, "news")

    @staticmethod
    def validDate(_year, _month, _day):
        # use datetime to figure out that date are available or not.
        try:
            _date = datetime(_year, _month, _day, 12, 0, 0)
        except ValueError:
            print("Invalid date")
            _date = None

        if _date is None:
            return None
        else:
            return "{}/{}/{}".format(_year, _month, _day)


class DateVector(NewsVector):
    def __init__(self):
        super().__init__()
        self.date_vector = []

    def wordVector(self, _word_vector, _year, _month, _day):
        _date = NewsVector.validDate(_year, _month, _day)

        # 若為無效日期
        if _date is None:
            return
        # 若為有效日期
        else:
            _dir_path = os.path.join(self.news_path, _date)

            # 若檔案不存在
            if not os.path.exists(_dir_path):
                return
            # 若檔案存在
            else:
                # 同一天中的所有新聞
                _files = os.listdir(_dir_path)
                # 遍歷所有新聞
                for _file in _files:
                    _file_vector = None
                    _path = os.path.join(_dir_path, _file)
                    print(_path)
                    # 開啟單篇新聞
                    with open(_path, "r", encoding="utf-8") as _content:
                        for _index, _line in enumerate(_content):
                            if _line == "" or _line == "\n":
                                continue

                            # 爬蟲結果可能一行中包含很多句子，要將其區分
                            _split_line = re.split("。| ？ | ! ", _line.strip("\n"))
                            # 將切開成多句的內容，依序做處理
                            for _sub_index, _sub_line in enumerate(_split_line):
                                if _sub_index == (len(_split_line) - 1):
                                    continue

                                # 將句子轉為向量 _sentence_vector.shape = (Ni, 250)
                                _sentence_vector = _word_vector.sentence2Vector(_sub_line)
                                # print("sentence_vector{}: {}".format(_index, _sentence_vector.shape))

                                # 將句向量 拼接進 檔案向量 _file_vector.shape = (N, 250)
                                if _file_vector is None:
                                    _file_vector = _sentence_vector.copy()
                                else:
                                    _file_vector = np.concatenate((_file_vector, _sentence_vector))

                        # 將檔案向量 拼接進 日向量 self.date_vector.shape = (file, N, 250)
                        self.date_vector.append(_file_vector)
                        print("file_vector:", np.array(_file_vector).shape)
                        print("date_vector:", np.array(self.date_vector).shape)

            return np.array(self.date_vector)

