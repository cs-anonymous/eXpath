for dataset in FB15k WN18 FB15k-237 WN18RR YAGO3-10; do
    for model in complex conve transe; do 
        output_dir="out/${model}_${dataset}"
        mkdir -p "$output_dir"
        python train.py --dataset "$dataset" --model "$model" > "${output_dir}/train.log"
    done
done

export CUDA_VISIBLE_DEVICES=0
for dataset in FB15k WN18 FB15k-237 WN18RR YAGO3-10; do
    output_dir="out/complex_${dataset}"
    mkdir -p "$output_dir"
    python train.py --dataset "$dataset" --model complex > "${output_dir}/train.log"
done

export CUDA_VISIBLE_DEVICES=1
for dataset in FB15k WN18 FB15k-237 WN18RR YAGO3-10; do
    output_dir="out/conve_${dataset}"
    mkdir -p "$output_dir"
    python train.py --dataset "$dataset" --model conve > "${output_dir}/train.log"
done


for dataset in FB15k WN18 FB15k-237 WN18RR YAGO3-10; do
    output_dir="out/transe_${dataset}"
    mkdir -p "$output_dir"
    python train.py --dataset "$dataset" --model transe > "${output_dir}/train.log"
done