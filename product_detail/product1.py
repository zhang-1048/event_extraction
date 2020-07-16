# coding = utf-8
import pandas as pd
import re
import math
import sys

sys.path.append("..")
from triple_extraction_file.functions_part import *
from triple_extraction_file.triple_extraction import TripleExtractor
import time
import numpy as np
import json
import exter_functions as efs

with open(file='../setting.json', mode='r', encoding='utf8')as fp:
    settings = json.load(fp)
    print('这是文件中的json数据：',settings)
    print('这是读取到文件数据的数据类型：', type(settings))

month = 6
upper_month = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二"]
# sheet_name = upper_month[month - 1] + "月"
sheet_name = "product_1"

start_time = "2020-0" + str(1) + "-01"
end_time = "2020-0" + str(month + 1) + "-01"

file_title_link = "./data/extraction_res/product_original_monthly" + str(month) + "_title_link_dict_monthly.npy"
file_event_link = "./data/extraction_res/product_original_monthly" + str(month) + "_event_link_dict_original_monthly.npy"
file_name = "../zongheinfo.xlsx"

data1 = pd.read_excel(file_name, sheet_name=sheet_name)
titles = list(data1['title'])
times = list(data1['pub_time'])
links = list(data1['url'])
ids = list(data1['id'])
keywords = list(data1['keywords'])
crawl_sites = list(data1['crawl_site'])
abstract_infos = list(data1['abstract_info'])
contents = list(data1['pure_content'])
pattern = re.compile("\d{4}/\d{1,2}/\d{1,2}")
meta_time = 1  # 事件间隔，周期==1天

start_time = pd.to_datetime(start_time)
end_time = pd.to_datetime(end_time)

dict_link_title = dict(zip(links, titles))
dict_link_ids = dict(zip(links, ids))
dict_link_contents = dict(zip(links, contents))
dict_link_title_original = dict(zip(links, titles))

dict_link_keywords = dict(zip(links, keywords))
dict_link_crawl_sites = dict(zip(links, crawl_sites))
dict_link_abstract_infos = dict(zip(links, abstract_infos))

print(len(links), len(titles))
dict_link_time = dict(zip(links, times))
data_dict = data_process(links=links, times=times, contents=contents,
                         pattern=pattern, meta_time=meta_time,
                         start_time=start_time, end_time=end_time)
triple_extractor = TripleExtractor()
k, v = list(data_dict.keys()), list(data_dict.values())
# print(len(k), len(v))


# total_count = 0
# for i in v:
#     total_count += len(i)
# print(total_count)
# efs.extraction_function(triple_extractor, k, v, file_event_link, total_count)

print("读取数据")
data_1 = np.load("./data/extraction_res/product_original_monthly.npy", allow_pickle=True)

specific_words = ["关键词"]
kw = open("../keywords.txt", "r", encoding="gbk", errors="ignore")
kw_list = kw.readlines()
kws = list(set([i.replace("\n", "") for i in kw_list]))
data_list = list(data_1) + list(data_2) + list(data_3)

original_event = efs.get_first_filter_process(data=data_list,
                                              kws=kws,
                                              specific_words=specific_words)

new_list = {}
total_list = []  # 记录全部数据的抽取结果
for k, v in original_event.items():
    total_list += v
    etp = []
    for it in v:
        etp.append("".join(it))
    new_list[k] = list(set(etp))
query_sent_list = new_list

title_allowed = True  # 是否允许标题也加入到结果中（主要针对新闻数据太少，信息捕获不足的情况）
filter_first, posttag_word_list = efs.filter_specific_word(query_sent_list=query_sent_list,
                                                           specific_words=specific_words,
                                                           dict_link_title_original=dict_link_title_original,
                                                           title_allowed=settings["title_allowed"])

print(len(total_list))
print(len(posttag_word_list))
print("加上词性标注的限制后，数量被筛选掉的数量：", len(total_list) - len(posttag_word_list))

# 开始构建规则
strage = ["关键词"]

operators = ["关键词"]

enterprise = ["关键词"]

zonghe_list, name_zonghe_list = efs.match_keyword_cls(strage=strage,
                                                      enterprise=enterprise,
                                                      operators=operators,
                                                      filter_first=filter_first)

res_file_name = "ZhiNengYingJian" + str(month) + ".xlsx"
efs.event_cluster_function(zonghe_list=zonghe_list,
                           name_zonghe_list=name_zonghe_list,
                           dict_link_title_original=dict_link_title_original,
                           dict_link_abstract_infos=dict_link_abstract_infos,
                           dict_link_time=dict_link_time,
                           dict_link_contents=dict_link_contents,
                           dict_link_ids=dict_link_ids,
                           dict_link_keywords=dict_link_keywords,
                           dict_link_crawl_sites=dict_link_crawl_sites,
                           res_file_name=res_file_name,
                           min_show_num=settings["min_show_num"],
                           max_show_num=settings["max_show_num"],
                           words_length_limit=settings["words_length_limit"],
                           remove_duplicated_link=settings["remove_duplicated_link"])
