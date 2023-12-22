export PYTHONPATH=${PWD}
for i in {0..9};
do
  torchrun --standalone --nnodes=1 --nproc_per_node=1 ./main/oecl/oecl.py --config_env ./yaml/oecl/env_oecl.yaml --config_exp ./yaml/oecl/oecl.yaml  --times 1 --seed $RANDOM --ddp True --id_class $i
done
