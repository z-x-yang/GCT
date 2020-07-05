import os
import json
import time
import shutil


ckpt_dir = './results/gct_resnet50'
ckpt_list_dir = os.path.join(ckpt_dir, "checkpoint")
temp_dir = './checkpoint.backup'

start_step = 0

shutil.move(ckpt_list_dir, temp_dir)

with open(temp_dir) as f:
	line = f.readline()
	while(line):
		if line[:3] == 'all':

			ckpt_index = line.split('\"')[-2]
			num_index = int(ckpt_index.split('-')[-1])
			print(ckpt_index)
			if num_index > start_step:
				with open(ckpt_list_dir, 'w') as f_ckpt:
					f_ckpt.write("model_checkpoint_path: \"" + ckpt_index + "\"")
				print("start eval: " + ckpt_index)
				output = os.popen('bash eval_gct_resnet50.sh').read()
				print(output)
		else:
			print('skip')
		line = f.readline()

shutil.move(temp_dir, ckpt_list_dir)