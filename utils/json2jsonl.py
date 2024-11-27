import json

def json_to_jsonl(json_file_path, jsonl_file_path):
    # 读取 json 文件
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)  # 假设这是个包含列表或字典的JSON文件

    # 检查数据是否是列表，如果不是，则将其转换为单元素列表
    if not isinstance(data, list):
        data = [data]

    # 将数据写入 jsonl 文件
    with open(jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:
        for item in data:
            jsonl_file.write(json.dumps(item, ensure_ascii=False))
            jsonl_file.write('\n')

# 使用方法
json_file_path = '/home/liuzhe/new-files/LoRA-XS/dataset-32-everypath.json'  # 输入的 JSON 文件路径
jsonl_file_path = '/home/liuzhe/new-files/LoRA-XS/dataset-32-everypath.jsonl'  # 输出的 JSONL 文件路径
json_to_jsonl(json_file_path, jsonl_file_path)