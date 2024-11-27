import json

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')

def process_questions(data):
    questions = []
    for item in data:
        if 'query' in item and ',' not in item['query']:
            questions.append(item['query'])
    return questions

def generate_combinations(questions):
    seen_combinations = set()
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            combo = f"{questions[i]}, {questions[j]}"
            if combo not in seen_combinations:
                seen_combinations.add(combo)
    # Convert the set back to a list of dictionaries
    return [{'question': combo, 'answer': "abc"} for combo in seen_combinations]

def main():
    # 假设输入文件为 input.json，输出文件为 output.jsonl
    input_file = '/home/liuzhe/new-files/LoRA-XS/utils/dataset-1024-everypath-9-7.json'
    output_file = '/home/liuzhe/new-files/LoRA-XS/utils/dataset-1024-everypath-9-7.jsonl'

    # 从 JSON 文件中加载数据
    data = read_json(input_file)

    # 处理问题并获取符合条件的问题列表
    valid_questions = process_questions(data)

    # 创建问题的组合
    question_combinations = generate_combinations(valid_questions)

    # 将组合写入新的 JSONL 文件
    write_jsonl(question_combinations, output_file)

if __name__ == "__main__":
    main()
