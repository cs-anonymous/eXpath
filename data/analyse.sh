
# list all directories in the dataset folder, ensure that it is directory
# folder=/home/sy/VIS/eXpath-demo/public/dataset/
datasets=$(ls -l . | grep ^d | awk '{print $9}')

# for each directory, list all files in the directory
for dataset in $datasets
do
    echo "Analyse $dataset"
    # python process_relation.py --dataset $dataset
    # python analyse_relation.py --dataset $dataset
    python analyse_entity.py --dataset $dataset
done
