# CIFAR-100 experiments

```bash
mkdir -p weights/cnn
mkdir -p results/cnn
mkdir -p bounds/cnn
weights_dir='weights/cnn'
output_json_dir='results/cnn'
bound_dir='bounds/cnn'

seeds=(
    7
    11
    13  
)

lr_list=(
    0.001
    0.0001
)

lambdas=(
    10
    100
    1000
    10000
    100000
    1000000
    10000000
    100000000
    1000000000
)

val_types=(
    "deterministic"
    "stochastic"
)
```

## supervised models

```bash
optimizers=(
    "sgd"
    "adam"
    "rmsprop"
)

for seed in "${seeds[@]}"
do
    for lr in "${lr_list[@]}"
    do
        for optimizer in "${optimizers[@]}"
        do
            python -m contrastive.supervised.supervised_cifar100 \
                --seed ${seed} \
                --lr ${lr} \
                --optim ${optimizer} \
                --output-model-name seed-${seed}_sup.pt
        done   
    done

    mkdir -p ${weights_dir}/sup/seed-${seed}
    mv *${seed}_sup* ${weights_dir}/sup/seed-${seed}
done

for seed in "${seeds[@]}"
do
    python -m contrastive.eval.top_k_run \
        --seed ${seed} \
        --output-json-fname ${output_json_dir}/sup-top-${seed}.json  \
        --model-name-dir ${weights_dir}/sup/seed-${seed} \
        --supervised

    python -m contrastive.eval.avg_run \
        --seed ${seed} \
        --output-json-fname ${output_json_dir}/sup-avg-${seed}.json \
        --model-name-dir ${weights_dir}/sup/seed-${seed} \
        --supervised 
done
```

---

## Deterministic models by Arora et al.

```
optimizers=(
    "sgd"
    "adam"
    "rmsprop"
)

for seed in "${seeds[@]}"
do
    for lr in "${lr_list[@]}"
    do
        for optimizer in "${optimizers[@]}"
        do
            python -m contrastive.cnn_run \
                --seed ${seed} \
                --lr ${lr} \
                --optim ${optimizer} \
                --output-model-name seed-${seed}_logistic.pt
        done
    done

    mkdir -p ${weights_dir}/arora/seed-${seed}
    mv *${seed}* ${weights_dir}/arora/seed-${seed}
done

for seed in "${seeds[@]}"
do
    python -m contrastive.eval.top_k_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/arora/seed-${seed} \
        --output-json-fname ${output_json_dir}/arora-top-${seed}.json

    python -m contrastive.eval.avg_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/arora/seed-${seed} \
        --output-json-fname ${output_json_dir}/arora-avg-${seed}.json 
done
```

---

## Model minimising PAC-Bayes objective with early stopping whose criterion is validation loss

```
optimizers=(
    "adam"
    "rmsprop"
)

for seed in "${seeds[@]}"
do
    for lr in "${lrs[@]}"
    do
        for optimizer in "${optimizers[@]}"
        do            
            for lambda in "${lambdas[@]}"
            do
                python -m contrastive.pb_cnn_run \
                    --seed ${seed} \
                    --lr ${lr} \
                    --optim ${optimizer} \
                    --catoni-lambda ${lambda} \
                    --output-model-name seed-${seed}_${lambda}.pt
            done
        done
    done
done

for val_type in "${val_types[@]}"
do
    for seed in "${seeds[@]}"
    do
        mkdir -p ${weights_dir}/${val_type}/seed-${seed}
        mv *${val_type}*${seed}* ${weights_dir}/${val_type}/seed-${seed}
    done
done

# evaluation
for seed in "${seeds[@]}"
do
    # deterministic
    python -m contrastive.eval.pb_top_k_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/deterministic/seed-${seed} \
        --output-json-fname ${output_json_dir}/deterministic-top-${seed}.json \
        --deterministic

    python -m contrastive.eval.pb_avg_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/deterministic/seed-${seed} \
        --output-json-fname ${output_json_dir}/deterministic-avg-${seed}.json \
        --deterministic 
    
    # stochastic
    python -m contrastive.eval.pb_top_k_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/stochastic/seed-${seed} \
        --output-json-fname ${output_json_dir}/stochastic-top-${seed}.json

    python -m contrastive.eval.pb_avg_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/stochastic/seed-${seed} \
        --output-json-fname ${output_json_dir}/stochastic-avg-${seed}.json
done

# calculatw PAC-bayes bounds inngredients
for seed in "${seeds[@]}"
do
    # deterministic
    python -m contrastive.eval.precompute_bound \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/deterministic/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}-deterministic-det.json \
        --deterministic 

    python -m contrastive.eval.precompute_bound \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/deterministic/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}-deterministic.json

    # stochastic
    python -m contrastive.eval.precompute_bound \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/stochastic/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}-stochastic-det.json \
        --deterministic
    python -m contrastive.eval.precompute_bound \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/stochastic/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}-stochastic.json
done
```

--- 

## PAC-Bayes Objective
```
optimizers=(
    "adam"
    "rmsprop"
)

for seed in "${seeds[@]}"
do
    for lr in "${lrs[@]}"
    do
        for optimizer in "${optimizers[@]}"
        do
            for lambda in "${lambdas[@]}"
            do
                python -m contrastive.pb_cnn_run \
                    --seed ${seed} \
                    --lr ${lr} \
                    --optim ${optimizer} \
                    --catoni-lambda ${lambda} \
                    --output-model-name seed-${seed}_${lambda}.pt \
                    --criterion pb \
                    --validation-ratio 0.
            done
        done
    done

    mkdir -p ${weights_dir}/pac-bayes/seed-${seed}
    mv *pb*${seed}* ${weights_dir}/pac-bayes/seed-${seed}
done

for seed in "${seeds[@]}"
do
    python -m contrastive.eval.precompute_bound \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/pac-bayes/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}-det.json \
        --criterion pb \
        --validation-ratio 0. \
        --deterministic
    python -m contrastive.eval.precompute_bound \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/pac-bayes/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}.json \
        --criterion pb \
        --validation-ratio 0.

    python -m contrastive.eval.pb_top_k_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/pac-bayes/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}.json \
        --output-json-fname ${output_json_dir}/pac-bayes-top-${seed}.json \
        --criterion pb \
        --validation-ratio 0.

    python -m contrastive.eval.pb_avg_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/pac-bayes/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}.json \
        --output-json-fname ${output_json_dir}/pac-bayes-avg-${seed}.json \
        --criterion pb \
        --validation-ratio 0.
done
```
