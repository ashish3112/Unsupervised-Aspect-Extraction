#!/usr/bin/env bash
# THEANO_FLAGS="device=gpu0,floatX=float32" python train.py \
# --emb ../preprocessed_data/restaurant/w2v_embedding \
#--domain restaurant \
# -o output_dir \
python3 train.py --emb ../../data/english/embeddings/crossling_english.vec.npy --domain restaurant --epochs 25 --aspect-size 14 --model-name _as14 --language english
python3 train.py --emb ../../data/english/embeddings/crossling_english.vec.npy --domain restaurant --epochs 25 --aspect-size 28 --model-name _as28 --language english
python3 train.py --emb ../../data/english/embeddings/crossling_english.vec.npy --domain restaurant --epochs 25 --aspect-size 42 --model-name _as42 --language english

python3 train.py --emb ../../data/finnish/embeddings/crossling_finnish.vec.npy --domain restaurant --epochs 25 --aspect-size 14 --model-name _as14 --language finnish
python3 train.py --emb ../../data/finnish/embeddings/crossling_finnish.vec.npy --domain restaurant --epochs 25 --aspect-size 28 --model-name _as28 --language finnish
python3 train.py --emb ../../data/finnish/embeddings/crossling_finnish.vec.npy --domain restaurant --epochs 25 --aspect-size 42 --model-name _as42 --language finnish

python3 train.py --emb ../../data/fi_en/embeddings/crossling_fi_en.vec.npy --domain restaurant --epochs 25 --aspect-size 14 --model-name _as14 --language fi_en
python3 train.py --emb ../../data/fi_en/embeddings/crossling_fi_en.vec.npy --domain restaurant --epochs 25 --aspect-size 28 --model-name _as28 --language fi_en
python3 train.py --emb ../../data/fi_en/embeddings/crossling_fi_en.vec.npy --domain restaurant --epochs 25 --aspect-size 42 --model-name _as42 --language fi_en
