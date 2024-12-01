import os


relation_set = set()
with open('train.txt', 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
print(len(lines))

for line in lines:
    if line:
        parts = line.strip().split('\t')
        relation_set.add(parts[1])

print(len(relation_set))
with open('relations.dict', 'w') as f:
    f.write('\n'.join([str(ix) + '\t' + r for ix, r in enumerate(relation_set)]))
