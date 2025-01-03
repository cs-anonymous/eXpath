# prepare data
for model in complex conve transe; do
    for dataset in FB15k WN18 FB15k-237 WN18RR; do
        mkdir -p out/KGEAttack/${model}_${dataset}
        # filename = model.lower() + dataset.lower().replace('-237', '237') + '_random.csv'
        filename=$(echo "${model}" | tr '[:upper:]' '[:lower:]')_$(echo "${dataset}" | tr '[:upper:]' '[:lower:]' | sed 's/-237/237/')"_random.csv"
        echo "$filename"

        cp input_facts/${filename} out/KGEAttack/${model}_${dataset}/target.txt
    done
done
    
# run attack
for model in complex conve transe; do
    for dataset in FB15k WN18 FB15k-237 WN18RR; do
        java -cp KGEATTACK.jar x.y.z.attack.Explain out/KGEAttack/${model}_${dataset}  data/${dataset}
    done
done
