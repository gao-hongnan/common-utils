# qsub -I -l select=1:ncpus=4:mem=64gb -P 11003281 -l walltime=24:00:00 -q normal

import os
import subprocess

import torch
import torch.distributed as dist
from torch.distributed import _shard

# Get IP address
process = subprocess.Popen(['hostname', '-i'], stdout=subprocess.PIPE)
stdout, _ = process.communicate()
ip_address = stdout.decode('utf-8').strip()

# Set MASTER_ADDR environment variable
os.environ['MASTER_ADDR'] = ip_address

# Get random unused port
port_command = '''
comm -23 <(seq 1 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1
'''
process = subprocess.Popen(port_command, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
stdout, _ = process.communicate()
port = stdout.decode('utf-8').strip()

# Set MASTER_PORT environment variable
os.environ['MASTER_PORT'] = port


def custom_setstate(self, state):
    # Bypass the process group check
    self._sharded_tensor_id = None

    # Continue with the original logic
    self._local_shards, self._metadata, pg_state, self._sharding_spec, self._init_rrefs = state

# Replace the original __setstate__ method with the custom one
_shard.sharded_tensor.api.ShardedTensor.__setstate__ = custom_setstate

path = "/home/project/11003281/multi-node-gaohn/ep0-ba20500-rank0.pt"
# path = "/home/project/11003281/multi-node-gaohn/ep0-ba54000.pt"
print(path)
shard = torch.load(path, map_location='cpu')
print(shard['state']['model']['model.transformer.blocks.0.attn.Wqkv.weight'])
#print(shard['state']['algorithms'])
#print(shard['state']['schedulers'])
#print(shard['state']['optimizers']["DecoupledAdamW"].keys())
#print(shard['state']['optimizers']["DecoupledAdamW"]["param_groups"])

