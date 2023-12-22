export PYTHONPATH=${PWD}
for i in 0;
do
  torchrun --standalone --nnodes=1 --nproc_per_node=1 ./main/oecl_self/oecl.py --config_env ./yaml/oecl_self/env.yaml --config_exp ./yaml/oecl_self/oecl.yaml  --times 1 --seed $RANDOM --ddp True --id_class $i
done
