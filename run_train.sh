# train model on 5 datasets and 3 models
for dataset in FB15k WN18 FB15k-237 WN18RR YAGO3-10; do
    for model in complex conve transe; do 
        output_dir="out/${model}_${dataset}"
        mkdir -p "$output_dir"
        python train.py --dataset "$dataset" --model "$model" > "${output_dir}/train.log"
    done
done
