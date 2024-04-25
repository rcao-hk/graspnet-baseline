python inference.py --split 'test_similar' --network_ver 'v0.6.4' --ckpt_epoch 46 --dump_dir 'ignet_v0.6.4' --gpu_id '0' --camera 'realsense' --voxel_size 0.002
python eval.py --split 'test_similar' --dump_dir 'ignet_v0.6.4'
# python inference_multimodal.py --split 'test_similar' --network_ver 'v0.8.0' --ckpt_epoch 48 --dump_dir 'ignet_v0.8.0' --gpu_id '2' --camera 'realsense' --voxel_size 0.002
# python eval.py --split 'test_similar' --dump_dir 'ignet_v0.8.0'