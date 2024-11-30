#!/bin/bash
python render_image.py
# Train, sample, and evaluate for category "cat"
echo "Starting training for category 'cat'..."
python ./stroke_generation/train.py --category "cat" --save_dir_name "cat" --coordinate_length 150 --log_interval 10000
echo "Training for 'cat' completed!"

echo "Starting sampling for category 'cat'..."
python ./stroke_generation/sampling.py --ckpt_path "./stroke_generation/results/diffusion-ddpm-cat/last.ckpt" --save_dir "./stroke_generation/results/diffusion-ddpm-cat/samples_cat"
echo "Sampling for 'cat' completed!"

echo "Starting evaluation for category 'cat'..."
python run_eval.py --fdir1 "./stroke_generation/results/diffusion-ddpm-cat/samples_cat" --fdir2 "./test/cat" --save_dir "./stroke_generation/results/diffusion-ddpm-cat/samples_cat"
echo "Evaluation for 'cat' completed!"

# Train, sample, and evaluate for category "garden"
echo "Starting training for category 'garden'..."
python ./stroke_generation/train.py --category "garden" --save_dir_name "garden" --coordinate_length 150 --log_interval 10000
echo "Training for 'garden' completed!"

echo "Starting sampling for category 'garden'..."
python ./stroke_generation/sampling.py --ckpt_path "./stroke_generation/results/diffusion-ddpm-garden/last.ckpt" --save_dir "./stroke_generation/results/diffusion-ddpm-garden/samples_garden"
echo "Sampling for 'garden' completed!"

echo "Starting evaluation for category 'garden'..."
python run_eval.py --fdir1 "./stroke_generation/results/diffusion-ddpm-garden/samples_garden" --fdir2 "./test/garden" --save_dir "./stroke_generation/results/diffusion-ddpm-garden/samples_garden"
echo "Evaluation for 'garden' completed!"

# Train, sample, and evaluate for category "helicopter"
echo "Starting training for category 'helicopter'..."
python ./stroke_generation/train.py --category "helicopter" --save_dir_name "helicopter" --coordinate_length 150 --log_interval 10000
echo "Training for 'helicopter' completed!"

echo "Starting sampling for category 'helicopter'..."
python ./stroke_generation/sampling.py --ckpt_path "./stroke_generation/results/diffusion-ddpm-helicopter/last.ckpt" --save_dir "./stroke_generation/results/diffusion-ddpm-helicopter/samples_helicopter"
echo "Sampling for 'helicopter' completed!"

echo "Starting evaluation for category 'helicopter'..."
python run_eval.py --fdir1 "./stroke_generation/results/diffusion-ddpm-helicopter/samples_helicopter" --fdir2 "./test/helicopter" --save_dir "./stroke_generation/results/diffusion-ddpm-helicopter/samples_helicopter"
echo "Evaluation for 'helicopter' completed!"

echo "All tasks completed successfully!"
