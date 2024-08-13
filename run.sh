export CUDA_VISIBLE_DEVICES=0

python main.py --output './examples' \
                    --content './data/content' \
                    --style './data/style' \
                    --sty_guidance 1.2 \
                    --num_inference_steps 2 \
                    --tome
