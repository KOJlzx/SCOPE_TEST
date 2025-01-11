export CUDA_VISIBLE_DEVICES=$1

method=$2 # Support ALLKV, PyramidKV, SnapKV, H2O, StreamingLLM
max_capacity_prompts=$3
attn_implementation=$4 # Support "flash_attention_2", "sdpa", "eager".
source_path=$5
model_path=$6
decoding_metric=$7 # H2O Support None,h2o,(fixed,linear,jump)---SCOPE
decoding_window_size=$8
save_dir=$9 # path to result save_dir
K=$10 #30,60
T=$11

python3 run_longgenbench.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --use_cache True \
    --K ${K}\
    --decoding_window_size ${decoding_window_size} \
    --decoding_recent_size ${decoding_recent_size} \
    --decoding_metric ${decoding_metric} \
    --max_num_examples ${T} \




CUDA_VISIBLE_DEVICES=0 python run_longgenbench.py --method ALLKV --model_path  meta-llama/Llama-3.2-1B --max_capacity_prompts 4096 --attn_implementation flash_attention_2 --save_dir results --use_cache True --K 30 --decoding_metric fixed 
python run_longgenbench.py --method ALLKV --model_path  meta-llama/Llama-3.2-1B --max_capacity_prompts 4096 --attn_implementation flash_attention_2 --save_dir results --use_cache True --K 30 
python run_longgenbench.py --method SnapKV --model_path  meta-llama/Llama-3.2-1B --max_capacity_prompts 4096 --attn_implementation flash_attention_2 --save_dir results --use_cache True --K 30 
python run_longgenbench.py --method ALLKV --model_path  mistralai/Mistral-7B-Instruct-v0.2 --max_capacity_prompts 4096 --attn_implementation flash_attention_2 --save_dir results --use_cache True --K 30 
