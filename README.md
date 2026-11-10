# pilot_v2 — Skill 학습 에이전트의 내재화·일반화·은퇴 (SkillRL 공개 체크포인트 평가)

> 질문: Skill을 프롬프트에 둔 채 RL로 학습한 에이전트는, 학습한 데이터셋에서 Skill 없이도 잘한다.
> 그 "없이도 된다"가 같은 절차의 다른 데이터셋에서도 유지되는가? (유지되지 않으면 Skill 은퇴는 위험하다)

## 현재 상태 (2026-11-10)

| 단계 | 내용 | 상태 |
|---|---|---|
| Step 1 | ALFWorld seen에서 **일반 Skill**(일반 6 + 실수 5)을 빼도 성능이 그대로인가 | **통과** — full 0.824 vs nogen 0.812, 효과 +0.012 [−0.024, +0.048], seed 3 |
| 바닥 확인 | 다른 데이터셋 원본 설정에서 RL이 바닥(<0.10)·천장(>0.90)이 아닌가 | 진행 중 — 원본+이력2: TWC 0.05, CookingWorld 0.00 (바닥), ScienceWorld 재실행 필요 → 원본+이력5 시도 중 |
| Step 2 | 사용 판정된 데이터셋에서 일반 Skill 넣기/빼기 (RL + 미학습 calib 대조) | 대기 |
| Step 3 | 지식 vs 절차 해석 (궤적 분석) | 대기 |
| Step 4 | 두 번째 도메인 Search (SkillRL Search 체크포인트) | 대기 |
| Step 5 | 안전한 은퇴 기준 비교 (이어 학습) | 대기 |

설계 메모는 `DESIGN_v2.md`, 측정 정의는 `MEASUREMENTS.md`, 실행 세부는 `RUNBOOK.md`.

## 구조

```
common/              프롬프트(SkillRL 템플릿 그대로), vLLM/mock 생성기, SkillRL SkillsOnlyMemory 로더
alfworld_eval/
  games.py           ALFWorld 게임 열거 (train/seen/unseen)
  perturb.py         표면 변형 (using / moveto / walkto / fromtake / names) + 되돌리기
  run_alfworld_eval.py  ALFWorld 한 조건 실행 (Skill bank, 변형, 검색 방식, seed)
  metrics.py         ALFWorld 집계 (A~G 지표)
envs/adapters.py     ScienceWorld · TWC · CookingWorld 어댑터 (동일 인터페이스, 난이도 옵션)
run_eval.py          다른 데이터셋 한 조건 실행
metrics_transfer.py  다른 데이터셋 집계
skills/
  sources.py         Skill 소스: none / general(일반 6+실수 5 원문) / procedural / mismatch_proc / skillrl_raw
  procedural_skills.json, procedure_map.json   (보조) 환경 무관 절차 Skill과 태그 맵
configs/
  names_map_example.json     이름 동의어 13개
  skills_mismatched.json     과제별 Skill을 다른 유형으로 바꾼 bank
  skills_nogeneral.json      일반·실수 Skill을 비운 bank (Step 1의 B 조건)
analysis/general_report.py   Step 1·2 보고서 (효과, CI, seed 간 잡음, 유형별)
necessity/           (보류) 임베딩 검색 자리 점유 실험
scripts/
  setup_workspace.sh  설치: uv 가상환경, 패키지, SkillRL·ALFWorld 데이터, 모델 다운로드, RL 병합(torch 2.8 전용 venv), Java
  env.sh              경로 변수 + 가상환경 활성화
  run_general.sh      Step 1 / 바닥 확인 / Step 2 실행기 (환경변수로 조건 지정)
  run_all.sh          (1차 실험용) 표면 변형·Skill 조건 일괄 실행
  smoke.sh            ALFWorld 6게임 스모크
  merge_fsdp_ckpt.py  verl FSDP 체크포인트 → HF safetensors 병합
  make_skill_variants.py      mismatched / nogeneral bank 생성
train/               연구실 GPU 2장용 SkillRL 이어 학습 스크립트와 안내 (Step 5용)
```

## 설치

```bash
export WORK=/root            # 연구실 서버면 /dataset/kjs1214 처럼 넉넉한 디스크
bash scripts/setup_workspace.sh
```

Python 3.12 이하 가상환경(`$WORK/venv`), vLLM, TextWorld/ALFWorld/ScienceWorld/TextWorldExpress, SkillRL 코드, ALFWorld 데이터,
SFT·RL 체크포인트(약 46GB) 다운로드와 RL 병합까지 한 번에. 끝나면 `cuda available: True`, `[setup] done.`.
Java(ScienceWorld용)는 `apt`가 없으면 경고만 내고 넘어감.

## 실행 (scripts/run_general.sh)

모든 조건은 환경변수로 지정. 끝난 조건은 건너뛰므로 같은 명령을 다시 걸면 이어서 돈다.

**Step 1 — ALFWorld, 일반 Skill 빼기 (RL, seed 3)**
```bash
source scripts/env.sh
SEEDS="0 1 2" MODELS="rl" SPLITS="seen" TRANSFER_ENVS="" OUT=runs_step1 bash scripts/run_general.sh
python analysis/general_report.py runs_step1        # (1) 섹션
```

**바닥 확인 — 다른 데이터셋 원본 설정, Skill 없음 (RL, seed 1)**
```bash
SEEDS="0" MODELS="rl" SPLITS="" TRANSFER_SKILLS="none" TRANSFER_VARS=20 \
  MAX_ACTIONS=0 MAX_STEPS=100 HISTORY=5 MAX_MODEL_LEN=16384 SW_SIMPL="" \
  TWC_PARAMS="numLocations=3, numItemsToPutAway=2, includeDoors=0, limitInventorySize=0" \
  CW_PARAMS="numLocations=3, numIngredients=2, numDistractorItems=5, includeDoors=0, limitInventorySize=0" \
  OUT=runs_floor_orig_h5 bash scripts/run_general.sh
python metrics_transfer.py "runs_floor_orig_h5/transfer/*.jsonl"
```

**Step 2 — 사용 판정된 데이터셋에서 일반 Skill 넣기/빼기 (RL + calib, seed 3)**
```bash
SEEDS="0 1 2" MODELS="rl calib" SPLITS="seen" TRANSFER_ENVS="twc" TRANSFER_VARS=20 \
  (위와 같은 난이도 옵션) OUT=runs_step2 bash scripts/run_general.sh
python analysis/general_report.py runs_step2
```
`SPLITS="seen"`을 두면 calib의 ALFWorld 넣기/빼기(미학습 모델 대조)도 같이 돈다.

### 환경변수

| 변수 | 기본값 | 뜻 |
|---|---|---|
| SEEDS | `0 1 2` | 반복 seed (seed마다 전체 게임 1회) |
| MODELS | `rl calib` | rl / calib / base |
| SPLITS | `seen unseen` | ALFWorld 분할. `""`이면 ALFWorld 생략 |
| TRANSFER_ENVS | `twc cookingworld scienceworld` | 다른 데이터셋. `""`이면 생략 |
| TRANSFER_SKILLS | `none general` | 다른 데이터셋 Skill 조건 |
| TRANSFER_VARS | 20 | 환경별 변형 수 (ScienceWorld는 과제당 test 변형 9개가 상한) |
| MAX_STEPS / MAX_ACTIONS / HISTORY | 50 / 150 / 2 | 스텝 상한 / 행동 목록 상한(0=무제한) / 이력 길이 |
| MAX_MODEL_LEN | 8192 | vLLM 최대 길이. 원본 ScienceWorld는 16384 필요 |
| SW_SIMPL | `noElectricalAction,openDoors,teleportAction` | ScienceWorld 단순화. `""`이면 원본 |
| TWC_PARAMS / CW_PARAMS | (방 1개 쉬운 설정) | TextWorldExpress 게임 설정. 원본은 위 예시 |
| OUT | `runs_gen` | 결과 폴더 |
| PARALLEL | 8 | 다른 데이터셋 동시 에피소드 수 |

## 결과 읽기

`analysis/general_report.py <OUT>`:
- **(1) ALFWorld**: `full`(Skill 그대로) vs `nogen`(일반 Skill 뺌) 성공률, 효과와 95% CI(게임 단위 부트스트랩, seed 평균), seed 간 뒤집힘 비율(잡음), 유형별 효과
- **(2) 다른 데이터셋**: `general − none` 효과 (성공률, 점수)
- 판정 기준(사전 고정): Step 1 통과 = 효과 ≤ +0.03 이고 CI 윗값 ≤ +0.06. 데이터셋 사용 = RL `none`이 0.10~0.90.

`metrics_transfer.py "<OUT>/transfer/*.jsonl"`: 데이터셋 × 모델 × Skill 조건별 성공률·점수·ALFWorld식 명령 침입률.

## 고정 설정

- 모델: RL = SkillRL 공개 `Alfworld-7B-RL`(학습률 기록상 step 80 추정) 병합본, calib = Qwen3-30B-A3B(ALFWorld 미학습), base = Qwen2.5-7B-Instruct
- 프롬프트: SkillRL 템플릿 그대로 (첫 스텝 Skill 없음, 이력 N스텝, 가능한 행동 목록), 온도 0.4, 최대 512토큰
- 일반 Skill 정의: SkillRL bank의 `general_skills` 앞 6개 + `common_mistakes` 앞 5개 (template 모드가 실제로 주입하는 것)
- 다른 데이터셋에 넣는 일반 Skill은 SkillRL `format_for_prompt`와 글자 단위로 같은 텍스트

## 알려진 한계

- 공개 RL 체크포인트는 학습 중간(step 80/150 추정)일 수 있음
- 공개 SFT 체크포인트는 학습 데이터 형식이 실제 환경과 달라 비교에서 제외
- `moveto` 변형은 실제 놓기 명령(`move X to Y`)과 겹치므로 표면 변형으로 쓰지 않음 (`walkto` 사용)
- 1회 실행의 성공률은 0.1 이상 흔들릴 수 있어 모든 비교는 seed 3개 이상
