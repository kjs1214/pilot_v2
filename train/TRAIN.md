# 연구실 GPU 2장으로 SkillRL RL 직접 학습 (Skill bank 고정)

## 0. 서버 확인 (먼저)
```bash
nvidia-smi                 # GPU 종류·메모리, 비어 있는 2장 번호
free -g                    # RAM: 7B는 CPU offload 때문에 180GB 이상 권장
nproc                      # CPU 코어: 환경 워커 128개(16×8)가 CPU에서 돎
df -h ~                    # 체크포인트 하나에 약 90GB(가중치+옵티마이저), 10스텝마다 저장
```
- GPU가 80GB(A100/H100)가 아니면 7B는 어려움 → 3B 모델로 축소 검토
- RAM이 부족하면 offload 불가 → 마찬가지로 3B

## 1. 학습 환경 (SkillRL README 기준, 이 평가 코드의 venv와 분리)
```bash
conda create -n skillrl python=3.10 -y && conda activate skillrl
git clone https://github.com/aiming-lab/SkillRL.git ~/SkillRL && cd ~/SkillRL
pip install -r requirements.txt
pip install vllm==0.11.0
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir   # 맞는 wheel이 없으면 소스 빌드(오래 걸림)
pip install -e .
pip install alfworld gymnasium==0.29.1 stable-baselines3==2.6.0
alfworld-download
```

## 2. 시작 모델 고르기
- **이어 학습 (추천, 빠름)**: 공개 RL 체크포인트(step 80 추정)를 병합한 `~/ckpt/rl_merged`에서 시작. 옵티마이저는 새로 시작하고,
  KL 기준 모델도 이 시작점이 됨(논문에 명시). 70~100스텝 더 돌리면 원래 150스텝 근처.
- **처음부터**: `~/ckpt/sft/checkpoint-140`에서 시작, 150스텝. 시간이 두 배 이상이지만 학습 곡선 전체를 얻음.

## 3. 실행
```bash
conda activate skillrl
export SKILLRL_ROOT=~/SkillRL CUDA_VISIBLE_DEVICES=0,1          # 비어 있는 두 장
cd ~/pilot_v2/train
MODEL_PATH=~/ckpt/rl_merged EXP=cont_rl80 EPOCHS=80 nohup bash train_skillrl_2gpu.sh > ~/train_cont_rl80.log 2>&1 &
tail -f ~/train_cont_rl80.log
```
- 메모리 부족(OOM)이면: `GPU_MEM=0.35` 또는 `FREE_CACHE=True`로 다시
- 진행: 로그의 step 번호, `val/` 로 시작하는 검증 성공률(10스텝마다)
- 체크포인트: `~/ckpt/train/cont_rl80/global_step_N/actor/` (10스텝마다)

## 4. 체크포인트를 평가 코드로 재기
학습 환경(torch 2.8)에서는 병합기가 바로 돌아요.
```bash
python ~/pilot_v2/scripts/merge_fsdp_ckpt.py --local_dir ~/ckpt/train/cont_rl80/global_step_80/actor --target_dir ~/ckpt/rl_step160
cd ~/pilot_v2 && source scripts/env.sh
RL=~/ckpt/rl_step160 OUT=runs_step160 REPS=1 SURFACE_MODELS="rl" SKILLGAP_MODELS="rl" bash scripts/run_all.sh surface
```
학습에 GPU 2장을 쓰는 동안 평가는 남는 GPU에서, 또는 학습이 끝난 뒤에.

## 예상 (추정)
- H100 2장: 스텝당 30~40분 → 70~80스텝 이어 학습 1.5~2일, 처음부터 150스텝 3~4일
- A100 80GB 2장: 위의 약 1.5배
- 첫 몇 스텝의 스텝당 시간으로 다시 계산할 것
