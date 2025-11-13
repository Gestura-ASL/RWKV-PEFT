load_model="/home/karthikssalian/work/RWKV-PEFT/out/rwkv-29.pth"
proj_dir='/home/karthikssalian/work/RWKV-PEFT/out'
data_file=/home/karthikssalian/work/RWKV-PEFT/json2binidx_tool/data/sample_text_document

n_layer=12
n_embd=768

micro_bsz=8
epoch_save=5
epoch_steps=10000
epoch_begin=30
ctx_len=512
peft_config='{"r":8,"lora_alpha":32,"lora_dropout":0.05}'


python train.py --load_model $load_model \
--proj_dir $proj_dir --data_file $data_file \
--vocab_size 65536 \
--data_type binidx \
--n_layer $n_layer --n_embd $n_embd \
--ctx_len $ctx_len --micro_bsz $micro_bsz \
--epoch_steps $epoch_steps --epoch_count 200 --epoch_save $epoch_save --epoch_begin  $epoch_begin \
--lr_init 1e-5 --lr_final 1e-5 \
--accelerator gpu --precision bf16 \
--devices 1 --strategy deepspeed_stage_1 --grad_cp 1 \
--my_testing "x070" \
--peft lora --peft_config $peft_config
