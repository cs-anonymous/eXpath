import openai
import json
from tqdm import tqdm
from openai import OpenAI
import argparse

arg = argparse.ArgumentParser()
arg.add_argument('--dataset', type=str, default='FB15k-237')

dataset = arg.parse_args().dataset

# 设置OpenAI API Key
api_key = 'sk-proj-XXXXX'
client = OpenAI(api_key=api_key)
MAX_LENGTH = 20

# 读取文件内容
def read_file(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                data[parts[0]] = parts[1]
    return data

# ensure each row is no longer than {MAX_LENGTH} words,
# 使用OpenAI API将描述转化为短语
def transform_description(descriptions):
    # 使用OpenAI的ChatCompletion API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Generate natural language phrase for the given relationships. Ensure each row is concise, succinct, fluent, meaningful, and follows a subject-verb-object structure. Do not use punctuation and 'the'. Provide relative clauses if needed. Avoid using 'relate' or 'invlove', and output the same number of rows as the input. Do not include order and do not add empty rows."},
            {"role": "user", "content": '\n'.join(descriptions)}
        ]
    )
    # 提取生成的短语
    transformed_phrase = response.choices[0].message.content.strip()
    return [line.strip() for line in transformed_phrase.split('\n') if len(line)]


# 检查输出是否符合条件
def validate_output(transformed_phrase, original_length, previous_phrases):
    # 检查是否有10行，且没有重复
    if len(transformed_phrase) != original_length:
        return False
    for phrase in transformed_phrase:
        if phrase in previous_phrases:
            return False
    return True

# 主函数
def process_files(input_file, output_file):
    mid2relation = read_file(input_file)
    
    mid_data = {}
    previous_phrases = set()

    def try_transform(descriptions):
        try:
            transformed_phrase = transform_description(descriptions)
            if validate_output(transformed_phrase, len(descriptions), previous_phrases):
                for i, description in enumerate(descriptions):
                    mid_data[description] = transformed_phrase[i]
                    previous_phrases.add(transformed_phrase[i])
                return            
        except Exception as e:
            print(e)
            print("Retrying...")
            try_transform(descriptions)

    descriptions = []
    for mid, description in tqdm(mid2relation.items()):
        descriptions.append(description)

        if len(descriptions) == 10:
            try_transform(descriptions)
            descriptions = []
            # 写入中间结果
            with open(output_file, 'w', encoding='utf-8') as json_file:
                json.dump(mid_data, json_file, ensure_ascii=False, indent=4)
    
    # 处理不足10条的记录
    if descriptions:
        try_transform(descriptions)
    
    # 最终写入JSON文件
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(mid_data, json_file, ensure_ascii=False, indent=4)

# 使用示例
input_file = dataset + '/relations.dict'
output_file = dataset + '/relation_name.json'

process_files(input_file, output_file)
