#BGL
#%%
from tqdm import tqdm
import re
import pickle
import numpy
from sentence_transformers import SentenceTransformer

#%%
path = '/home/whut4/liyafei/dataset/BGL/'
name = 'BGL'
#%%
import re
import pickle

def extract_log_template(log_file):
    # 读取日志文件内容
    with open(log_file, "r") as file:
        log_contents = file.readlines()
    log_contents = log_contents
    # 提取日志模板
    log_template_set = set()
    log_template_indices = []
    block_size = 20
    block_array = []

    for i, log_line in enumerate(log_contents, start=1):
        log_line = log_line.strip()

        # 将数字和IP地址替换为占位符
        log_line = re.sub(r'\d+', '###', log_line)
        log_line = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '###', log_line)

        if log_line not in log_template_set:
            log_template_set.add(log_line)

        log_template_indices.append(list(log_template_set).index(log_line))

        # 每20条日志存储为小数组
        if i % block_size == 0:
            block_array.append(log_template_indices[-block_size:])

    return list(log_template_set), block_array



# 处理normal.log文件
normal_log_file = "normal.log"
normal_templates, normal_block_array = extract_log_template(path + normal_log_file)
print('Success normal')
# 处理abnormal.log文件
abnormal_log_file = "abnormal.log"
abnormal_templates, abnormal_block_array = extract_log_template(path + abnormal_log_file)
print('Success abnormal')


model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')


normal_embeddings = model.encode(normal_templates)
print('Success embedding')
savePath = path = '/home/whut4/liyafei/dataset/BGL/'
with open(savePath + "normal_templates.pickle", "wb") as file:
    pickle.dump(normal_embeddings, file)
# 存储normal.log每一条日志所对应的模板的索引
with open(savePath + "blknormal.pickle", "wb") as file:
    pickle.dump(normal_block_array, file)


abnormal_embeddings = model.encode(abnormal_templates)
print('Success embedding')
# 存储abnormal模板文件
with open(savePath + "abnormal_templates.pickle", "wb") as file:
    pickle.dump(abnormal_embeddings, file)
# 存储abnormal.log每一条日志所对应的模板的索引
with open(savePath + "blkabnormal.pickle", "wb") as file:
    pickle.dump(abnormal_block_array, file)