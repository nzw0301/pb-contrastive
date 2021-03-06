# AUSLAN Experiments

```bash
mkdir -p weights/mlp
mkdir -p results/mlp
mkdir -p bounds/mlp
weights_dir='weights/mlp'
output_json_dir='results/mlp'
bound_dir='bounds/mlp'
root='/home/YOUR_NAME/data/australian'

validation_ratio=0.125

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
    1
    10
    100
    1000
    10000
    100000
)
```

---

### Supervised

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
                --validation-ratio ${validation_ratio} \
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
        --validation-ratio ${validation_ratio} \
        --supervised

    python -m contrastive.eval.avg_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/sup/seed-${seed} \
        --output-json-fname ${output_json_dir}/sup-avg-${seed}.json \
        --root ${root} \
        --mlp \
        --dim-h 50 \
        --validation-ratio ${validation_ratio} \
        --supervised
done
```

---

### Algorithm proposed by Arora et al., 2019.

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
            python -m contrastive.mlp_run \
                --seed ${seed} \
                --lr ${lr} \
                --optim ${optimizer} \
                --output-model-name seed-${seed}_logistic.pt \
                --root ${root} \
                --dim-h 50 \
                --validation-ratio ${validation_ratio}
        done
    done
    mkdir -p ${weights_dir}/arora/seed-${seed}
    mv *${seed}_logistic\.* ${weights_dir}/arora/seed-${seed}
done


for seed in "${seeds[@]}"
do
    python -m contrastive.eval.top_k_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/arora/seed-${seed} \
        --output-json-fname ${output_json_dir}/arora-top-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --validation-ratio ${validation_ratio}

    python -m contrastive.eval.avg_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/arora/seed-${seed} \
        --output-json-fname ${output_json_dir}/arora-avg-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --validation-ratio ${validation_ratio}
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
    for lr in "${lr_list[@]}"
    do
        for optimizer in "${optimizers[@]}"
        do
            for lambda in "${lambdas[@]}"
            do
                python -m contrastive.pb_mlp_run \
                    --seed ${seed} \
                    --lr ${lr} \
                    --optim ${optimizer} \
                    --catoni-lambda ${lambda} \
                    --output-model-name seed-${seed}_mlp_${lambda}.pt \
                    --root ${root} \
                    --dim-h 50 \
                    --validation-ratio ${validation_ratio}

            done
        done
    done

    mkdir -p ${weights_dir}/stochastic/seed-${seed}
    mv lr*stochastic*${seed}_mlp* ${weights_dir}/stochastic/seed-${seed}
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
        --validation-ratio ${validation_ratio} \
        --deterministic

    python -m contrastive.eval.pb_top_k_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/stochastic/seed-${seed} \
        --output-json-fname ${output_json_dir}/stochastic-top-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --validation-ratio ${validation_ratio}

    python -m contrastive.eval.pb_avg_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/deterministic/seed-${seed} \
        --output-json-fname ${output_json_dir}/deterministic-avg-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --validation-ratio ${validation_ratio} \
        --deterministic

    python -m contrastive.eval.pb_avg_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/stochastic/seed-${seed} \
        --output-json-fname ${output_json_dir}/stochastic-avg-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --validation-ratio ${validation_ratio}
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
        --validation-ratio ${validation_ratio} \
        --deterministic

    python -m contrastive.eval.precompute_bound \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/deterministic/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}-deterministic.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --validation-ratio ${validation_ratio}

    # stochastic
    python -m contrastive.eval.precompute_bound \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/stochastic/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}-stochastic-det.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --validation-ratio ${validation_ratio} \
        --deterministic

    python -m contrastive.eval.precompute_bound \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/stochastic/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}-stochastic.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --validation-ratio ${validation_ratio}
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
    for lr in "${lr_list[@]}"
    do
        for optimizer in "${optimizers[@]}"
        do
            for lambda in "${lambdas[@]}"
            do
                python -m contrastive.pb_mlp_run \
                    --seed ${seed} \
                    --lr ${lr} \
                    --optim ${optimizer} \
                    --catoni-lambda ${lambda} \
                    --output-model-name seed-${seed}_mlp_${lambda}.pt \
                    --root ${root} \
                    --dim-h 50 \
                    --validation-ratio 0. \
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
        --validation-ratio 0. \
        --criterion pb

    python -m contrastive.eval.precompute_bound \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/pac-bayes/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}-det.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --validation-ratio 0. \
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
        --validation-ratio 0. \
        --criterion pb

    python -m contrastive.eval.pb_avg_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/pac-bayes/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}.json \
        --output-json-fname ${output_json_dir}/pac-bayes-avg-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --validation-ratio 0. \
        --criterion pb
done
```
