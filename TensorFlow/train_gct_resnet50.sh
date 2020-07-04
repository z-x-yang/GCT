DATA_DIR="/path/to/imagenet"
CKPT_DIR="results/gct_resnet50"
echo ${CKPT_DIR}
python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=64 \
--model=resnet50 --optimizer=momentum --variable_update=replicated \
--nodistortions --gradient_repacking=8 --num_gpus=4 \
--num_epochs=100 --weight_decay=1e-4 --data_dir=${DATA_DIR} \
--train_dir=${CKPT_DIR} --print_training_accuracy --xla --save_model_secs=3600 \
--summary_verbosity=1 --save_summaries_steps=200 --eval_dir=${CKPT_DIR}/eval