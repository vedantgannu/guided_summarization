# INPUT=/path/to/the/input/file
# OUTPUT=/path/to/the/output/file
INPUT=./cnn_dm/val.source
OUTPUT=./cnn_dm/input_dir/val.bpe.source
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
# python -m fairseq_orig.fairseq.examples.roberta.multiprocessing_bpe_encoder \
python -m fairseq.examples.roberta.multiprocessing_bpe_encoder \
--encoder-json encoder.json \
--vocab-bpe vocab.bpe \
--inputs "$INPUT" \
--outputs "$OUTPUT" \
--workers 60 \
--keep-empty;
