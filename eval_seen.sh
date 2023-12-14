python inference.py --split 'test_seen' --camera 'realsense' --network_ver 'v0.3.3.5' --dump_dir 'ignet_v0.3.3.5' --gpu_id '0'
python eval.py --split 'test_seen' --dump_dir 'ignet_v0.3.3.5'