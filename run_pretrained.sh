#!/bin/bash
python render_image.py

echo "Starting sampling for category 'cat'..."
python ./stroke_generation/sampling.py --ckpt_path "./pretrained/last_cat_150.ckpt" --save_dir "./pretrained/results/samples_cat"
echo "Sampling for 'cat' completed!"

echo "Starting evaluation for category 'cat'..."
python run_eval.py --fdir1 "./pretrained/results/samples_cat" --fdir2 "./test/cat" --save_dir "./pretrained/results/samples_cat"
echo "Evaluation for 'cat' completed!"

echo "Starting sampling for category 'garden'..."
python ./stroke_generation/sampling.py --ckpt_path "./pretrained/last_garden_150.ckpt" --save_dir "./pretrained/results/samples_garden"
echo "Sampling for 'garden' completed!"

echo "Starting evaluation for category 'garden'..."
python run_eval.py --fdir1 "./pretrained/results/samples_garden" --fdir2 "./test/garden" --save_dir "./pretrained/results/samples_garden"
echo "Evaluation for 'garden' completed!"

echo "Starting sampling for category 'helicopter'..."
python ./stroke_generation/sampling.py --ckpt_path "./pretrained/last_helicopter_150.ckpt" --save_dir "./pretrained/results/samples_helicopter"
echo "Sampling for 'helicopter' completed!"

echo "Starting evaluation for category 'helicopter'..."
python run_eval.py --fdir1 "./pretrained/results/samples_helicopter" --fdir2 "./test/helicopter" --save_dir "./pretrained/results/samples_helicopter"
echo "Evaluation for 'helicopter' completed!"

echo "All tasks completed successfully!"
