import math

with open('output_end_to_end.csv', 'r') as f:
    lines = f.readlines()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

output = open('output.txt', 'w')
for line in lines:
    parts = line.split(';')

    output.write(';'.join(parts[:3]) + '\n')
    relevance = int(parts[-1]) - int(parts[-2]) + sigmoid(float(parts[-4]) - float(parts[-3]))
    output.write(';'.join(parts[3:6]) + ':' + str(relevance) + '\n\n')
    
output.close()