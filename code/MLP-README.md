## MLP Experiments

```bash
mkdir -p weights/mlp
mkdir -p results/mlp
mkdir -p bounds/mlp
weights_dir='weights/mlp'
output_json_dir='results/mlp'
bound_dir='bounds/mlp'
root='/home/YOUR_NAME/data/australian'

seeds=(
    7
    11
    13
)

lrs=(
    0.001
    0.0001
)

lambdas=(
    1
    10
    100
    1000
    10000
    100000
)
```

### supervised 

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
            python -m contrastive.supervised.supervised_mlp \
                --seed ${seed} \
                --lr ${lr} \
                --optim ${optimizer} \
                --output-model-name seed-${seed}_sup.pt \
                --root ${root}
        done   
    done
    mkdir -p ${weights_dir}/sup/seed-${seed}
    mv *${seed}_sup* ${weights_dir}/sup/seed-${seed}
done

for seed in "${seeds[@]}"
do
    python -m contrastive.eval.top_k_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/sup/seed-${seed} \
        --output-json-fname ${output_json_dir}/sup-top-${seed}.json \
        --root ${root} \
        --mlp \
        --dim-h 50 \
        --supervised

    python -m contrastive.eval.avg_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/sup/seed-${seed} \
        --output-json-fname ${output_json_dir}/sup-avg-${seed}.json \
        --root ${root} \
        --mlp \
        --dim-h 50 \
        --supervised
done
```

### Algorithm proposed by Arora et al.

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
            python -m contrastive.mlp-run \
                --seed ${seed} \
                --lr ${lr} \
                --optim ${optimizer} \
                --output-model-name seed-${seed}_logistic.pt \
                --root ${root} \
                --dim-h 50
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
        --output-json-fname ${output_json_dir}/arora-top-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50

    python -m contrastive.eval.avg_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/arora/seed-${seed} \
        --output-json-fname ${output_json_dir}/arora-avg-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50
done
```

---

## Stochastic models with early stopping

```bash
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
                python -m contrastive.pb-mlp-run \
                    --seed ${seed} \
                    --lr ${lr} \
                    --optim ${optimizer} \
                    --catoni-lambda ${lambda} \
                    --output-model-name seed-${seed}_mlp_${lambda}.pt \
                    --dim-h 50 \
                    --root ${root}
            done
        done
    done

    mkdir -p ${weights_dir}/stochastic/seed-${seed}
    mv lr*stochastic*${seed}mlp* ${weights_dir}/stochastic/seed-${seed}
    mkdir -p ${weights_dir}/deterministic/seed-${seed}
    mv lr*deterministic*${seed}_mlp* ${weights_dir}/deterministic/seed-${seed}
done

for seed in "${seeds[@]}"
do
    python -m contrastive.eval.pb_top_k_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/deterministic/seed-${seed} \
        --output-json-fname ${output_json_dir}/deterministic-top-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --deterministic

    python -m contrastive.eval.pb_top_k_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/stochastic/seed-${seed} \
        --output-json-fname ${output_json_dir}/stochastic-top-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50
    
    python -m contrastive.eval.pb_avg_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/deterministic/seed-${seed} \
        --output-json-fname ${output_json_dir}/deterministic-avg-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --deterministic

    python -m contrastive.eval.pb_avg_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/stochastic/seed-${seed} \
        --output-json-fname ${output_json_dir}/stochastic-avg-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50
done

# compute bounds of ingredients
for seed in "${seeds[@]}"
do
    # deterministic
    python -m contrastive.eval.precompute_bound \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/deterministic/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}-deterministic-det.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --deterministic

    python -m contrastive.eval.precompute_bound \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/deterministic/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}-deterministic.json \
        --mlp \
        --root ${root} \
        --dim-h 50

    # stochastic    
    python -m contrastive.eval.precompute_bound \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/stochastic/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}-stochastic-det.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --deterministic
    python -m contrastive.eval.precompute_bound \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/stochastic/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}-stochastic.json \
        --mlp \
        --root ${root} \
        --dim-h 50
done
```

---

## Stochastic models without early stopping

```bash
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
                python -m contrastive.pb-mlp-run \
                    --seed ${seed} \
                    --lr ${lr} \
                    --optim ${optimizer} \
                    --catoni-lambda ${lambda} \
                    --output-model-name seed-${seed}_mlp_${lambda}.pt \
                    --root ${root} \
                    --dim-h 50 \
                    --criterion pb
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
        --json-fname ${bound_dir}/pac-bayes-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --criterion pb
    python -m contrastive.eval.precompute_bound \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/pac-bayes/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}-det.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --criterion pb \
        --deterministic

    python -m contrastive.eval.pb_top_k_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/pac-bayes/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}.json \
        --output-json-fname ${output_json_dir}/pac-bayes-top-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --criterion pb

    python -m contrastive.eval.pb_avg_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/pac-bayes/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}.json \
        --output-json-fname ${output_json_dir}/pac-bayes-avg-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --criterion pb
done
```
