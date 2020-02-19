## Non-iid Experiments

```bash
mkdir -p weights/non-iid
mkdir -p results/non-iid
mkdir -p bounds/non-iid
weights_dir='weights/non-iid'
output_json_dir='results/non-iid'
bound_dir='bounds/non-iid'
root='/home/YOUR_NAME/data/australian'

validation_ratio=0.125
prior_log_std=-2.5

seeds=(
    7
    11
    13
)

lr_list=(
    0.001
    0.0001
)
```

## Supervised

Same to `MLP-README.md`'s supervised section.

---

## Empirical risk minimisation (Algorithm proposed by Arora et al., 2019.)

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
                --output-model-name seed-${seed}_logistic_noniid.pt \
                --root ${root} \
                --dim-h 50 \
                --validation-ratio ${validation_ratio} \
                --non-iid
        done
    done
    mkdir -p ${weights_dir}/arora/seed-${seed}
    mv *${seed}_logistic_noniid* ${weights_dir}/arora/seed-${seed}
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
        --validation-ratio ${validation_ratio} \
        --non-iid

    python -m contrastive.eval.avg_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/arora/seed-${seed} \
        --output-json-fname ${output_json_dir}/arora-avg-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --validation-ratio ${validation_ratio} \
        --non-iid
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
            python -m contrastive.non_iid_pb_mlp_run \
                --seed ${seed} \
                --lr ${lr} \
                --optim ${optimizer} \
                --output-model-name seed-${seed}_noniid.pt \
                --dim-h 50 \
                --validation-ratio ${validation_ratio} \
                --root ${root} \
                --prior-log-std ${prior_log_std} \
                --non-iid
        done
    done

    mkdir -p ${weights_dir}/stochastic/seed-${seed}
    mv lr*stochastic*${seed}_noniid* ${weights_dir}/stochastic/seed-${seed}
    mkdir -p ${weights_dir}/deterministic/seed-${seed}
    mv lr*deterministic*${seed}_noniid* ${weights_dir}/deterministic/seed-${seed}
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
        --deterministic \
        --validation-ratio ${validation_ratio} \
        --non-iid

    python -m contrastive.eval.pb_top_k_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/stochastic/seed-${seed} \
        --output-json-fname ${output_json_dir}/stochastic-top-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --validation-ratio ${validation_ratio} \
        --non-iid

    python -m contrastive.eval.pb_avg_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/deterministic/seed-${seed} \
        --output-json-fname ${output_json_dir}/deterministic-avg-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --deterministic \
        --validation-ratio ${validation_ratio} \
        --non-iid

    python -m contrastive.eval.pb_avg_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/stochastic/seed-${seed} \
        --output-json-fname ${output_json_dir}/stochastic-avg-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --validation-ratio ${validation_ratio} \
        --non-iid
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
        --deterministic \
        --validation-ratio ${validation_ratio} \
        --non-iid \
        --non-iid-bound

    python -m contrastive.eval.precompute_bound \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/deterministic/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}-deterministic.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --validation-ratio ${validation_ratio} \
        --non-iid \
        --non-iid-bound

    # stochastic
    python -m contrastive.eval.precompute_bound \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/stochastic/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}-stochastic-det.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --deterministic \
        --validation-ratio ${validation_ratio} \
        --non-iid \
        --non-iid-bound

    python -m contrastive.eval.precompute_bound \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/stochastic/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}-stochastic.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --validation-ratio ${validation_ratio} \
        --non-iid \
        --non-iid-bound
done
```

---

## Stochastic models without early stopping

```bash
# PAC-Bayes
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
            python -m contrastive.non_iid_pb_mlp_run \
                --seed ${seed} \
                --lr ${lr} \
                --optim ${optimizer} \
                --output-model-name seed-${seed}_mlp_noniid.pt \
                --root ${root} \
                --dim-h 50 \
                --criterion pb \
                --validation-ratio 0. \
                --prior-log-std ${prior_log_std} \
                --non-iid
        done
    done
    mkdir -p ${weights_dir}/pac-bayes/seed-${seed}
    mv *pb*${seed}*_noniid* ${weights_dir}/pac-bayes/seed-${seed}
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
        --criterion pb \
        --validation-ratio 0. \
        --non-iid \
        --non-iid-bound

    python -m contrastive.eval.precompute_bound \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/pac-bayes/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}-det.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --criterion pb \
        --validation-ratio 0. \
        --deterministic \
        --non-iid \
        --non-iid-bound

    python -m contrastive.eval.pb_top_k_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/pac-bayes/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}.json \
        --output-json-fname ${output_json_dir}/pac-bayes-top-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --criterion pb \
        --validation-ratio 0. \
        --non-iid

    python -m contrastive.eval.pb_avg_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/pac-bayes/seed-${seed} \
        --json-fname ${bound_dir}/pac-bayes-${seed}.json \
        --output-json-fname ${output_json_dir}/pac-bayes-avg-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --criterion pb \
        --validation-ratio 0. \
        --non-iid
done
```

---

## Catoni's iid based algorithm with earlystopping

```bash
lambdas=(
    1
    10
    100
    1000
    10000
    100000
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
                    --output-model-name seed-${seed}_mlp_catoni_${lambda}.pt \
                    --dim-h 50 \
                    --validation-ratio ${validation_ratio} \
                    --root ${root} \
                    --non-iid
            done
        done
    done

    mkdir -p ${weights_dir}/catoni-stochastic/seed-${seed}
    mv lr*stochastic*${seed}_mlp_catoni_* ${weights_dir}/catoni-stochastic/seed-${seed}
    mkdir -p ${weights_dir}/catoni-deterministic/seed-${seed}
    mv lr*deterministic*${seed}_mlp_catoni_* ${weights_dir}/catoni-deterministic/seed-${seed}
done

for seed in "${seeds[@]}"
do
    python -m contrastive.eval.pb_top_k_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/catoni-deterministic/seed-${seed} \
        --output-json-fname ${output_json_dir}/catoni-deterministic-top-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --validation-ratio ${validation_ratio} \
        --deterministic \
        --non-iid

    python -m contrastive.eval.pb_top_k_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/catoni-stochastic/seed-${seed} \
        --output-json-fname ${output_json_dir}/catoni-stochastic-top-${seed}.json \
        --mlp \
        --root ${root} \
        --validation-ratio ${validation_ratio} \
        --dim-h 50 \
        --non-iid

    python -m contrastive.eval.pb_avg_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/catoni-deterministic/seed-${seed} \
        --output-json-fname ${output_json_dir}/catoni-deterministic-avg-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --validation-ratio ${validation_ratio} \
        --deterministic \
        --non-iid

    python -m contrastive.eval.pb_avg_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/catoni-stochastic/seed-${seed} \
        --output-json-fname ${output_json_dir}/catoni-stochastic-avg-${seed}.json \
        --mlp \
        --root ${root} \
        --validation-ratio ${validation_ratio} \
        --dim-h 50 \
        --non-iid
done
```

---

## Catoni's iid based algorithm without earlystopping

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
                    --output-model-name seed-${seed}_mlp_catoni_${lambda}.pt \
                    --root ${root} \
                    --dim-h 50 \
                    --validation-ratio 0. \
                    --criterion pb \
                    --non-iid
            done
        done
    done
    mkdir -p ${weights_dir}/catoni-pac-bayes/seed-${seed}
    mv *pb*${seed}_mlp_catoni_* ${weights_dir}/catoni-pac-bayes/seed-${seed}
done

for seed in "${seeds[@]}"
do
    python -m contrastive.eval.precompute_bound \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/catoni-pac-bayes/seed-${seed} \
        --json-fname ${bound_dir}/catoni-pac-bayes-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --validation-ratio 0. \
        --criterion pb \
        --non-iid

    python -m contrastive.eval.precompute_bound \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/catoni-pac-bayes/seed-${seed} \
        --json-fname ${bound_dir}/catoni-pac-bayes-${seed}-det.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --validation-ratio 0. \
        --criterion pb \
        --deterministic \
        --non-iid

    python -m contrastive.eval.pb_top_k_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/catoni-pac-bayes/seed-${seed} \
        --json-fname ${bound_dir}/catoni-pac-bayes-${seed}.json \
        --output-json-fname ${output_json_dir}/catoni-pac-bayes-top-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --validation-ratio 0. \
        --criterion pb

    python -m contrastive.eval.pb_avg_run \
        --seed ${seed} \
        --model-name-dir ${weights_dir}/catoni-pac-bayes/seed-${seed} \
        --json-fname ${bound_dir}/catoni-pac-bayes-${seed}.json \
        --output-json-fname ${output_json_dir}/catoni-pac-bayes-avg-${seed}.json \
        --mlp \
        --root ${root} \
        --dim-h 50 \
        --validation-ratio 0. \
        --criterion pb
done
```
