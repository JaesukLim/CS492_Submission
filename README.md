# CS492(D) Team 18 Submission

- 20200743 임재석
- 20200623 조홍훈
- 20220207 김형태

## Instruction
0. Install git-lfs and clone with lfs
```
sudo apt install git-lfs
git lfs clone https://github.com/JaesukLim/CS492_Submission
```
1. Install the dependencies
```
pip install -r requirements.txt
```

2. Use ```run_pretrained.sh``` for sampling and evaluating with pretrained model (coord_length: 150)
```
./run_pretrained.sh
```
If you want to train with different setting, use 
```
./run.sh
```

3. You can find FID and KID result from pretrained model in the directory below
```
./pretrained/results/samples_{CATEGORY}/result.json
```
You can find FID and KID evaluation result in the directory below if you used ```./run.sh```
```
./stroke_generation/results/diffusion-ddpm-{CATEGORY}/samples_{CATEGORY}/result.json
```