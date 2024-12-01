import json
import html


def read_txt(triples_path, separator="\t"):
    with open(triples_path, 'r') as file:
        lines = file.readlines()
    
    textual_triples = []
    for line in lines:
        line = html.unescape(line).lower()   # this is required for some YAGO3-10 lines
        head_name, relation_name, tail_name = line.strip().split(separator)

        # remove unwanted characters
        head_name = head_name.replace(",", "").replace(":", "").replace(";", "")
        relation_name = relation_name.replace(",", "").replace(":", "").replace(";", "")
        tail_name = tail_name.replace(",", "").replace(":", "").replace(";", "")

        textual_triples.append((head_name, relation_name, tail_name))
    return textual_triples

textual_triples = read_txt(f"train.txt")
id_list = set()
txt_file = open('entities.dict', 'w')

for h, _, t in textual_triples:
    id_list.add(h)
    id_list.add(t)

for ix, id in enumerate(id_list):
    txt_file.write(f"{ix}\t{id}\n")

txt_file.close()
