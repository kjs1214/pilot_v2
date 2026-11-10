# pilot_v2

Skill을 프롬프트에 넣어 RL로 학습한 에이전트는 학습한 데이터셋에서 Skill 없이도 잘한다.
**그 "없이도 된다"가 같은 절차의 다른 데이터셋에서도 유지되는지**를 재는 실험 코드다.
대상 모델은 SkillRL이 공개한 ALFWorld RL 체크포인트다.

---

## 1. 무엇을 판정하나

두 조건이 함께 성립하면 "일반화 실패"로 본다. 기준은 결과를 보기 전에 고정했다.

| | 어디서 | 조건 |
|---|---|---|
| ① | ALFWorld (학습한 데이터셋) | 일반 지침을 넣어도 차이가 없다: 효과 ≤ +0.03, CI 윗값 ≤ +0.06 |
| ② | 다른 데이터셋 | 일반 지침을 넣으면 오른다: 효과 ≥ +0.10, CI 아랫값 > 0 |

- 판정은 **11개 조건**으로 하고, 6개 조건은 함께 보고한다.
- 넣어도 0.10 미만이면 "이 설정에서는 판정 불가"로 보고한다. 설정을 쉽게 바꿔 다시 재지 않는다.
- ScienceWorld는 성공률이 아니라 벤치마크의 진행 점수(0~1)로 판정한다.

## 2. 지금까지의 결과

| 단계 | 내용 | 결과 |
|---|---|---|
| Step 1 | ALFWorld에서 과제별 Skill은 두고 일반 지침만 빼기 (RL, seed 3) | 통과. seen +0.012 [−0.024, +0.048], unseen +0.017 [−0.025, +0.060] |
| Step 1 대조 | 같은 실험을 미학습 모델(calib)로 | 11개를 빼면 +0.050 [+0.014, +0.086] 하락. 쓸모 있는 지침이다 |
| Step 2 | 세 단계 조건을 ALFWorld와 다른 데이터셋에서 | 진행 중 |

## 3. 세 단계 조건

| | Skill 없음 | 일반 6만 | 11개 (일반 6 + 실수 5) |
|---|---|---|---|
| ALFWorld (`BANKS`) | `nosk` | `genonly6` | `genonly` |
| 다른 데이터셋 (`TRANSFER_SKILLS`) | `none` | `general6` | `general` |

- "일반 6"은 SkillRL bank의 `general_skills` 앞 6개, "실수 5"는 `common_mistakes` 앞 5개다. SkillRL이 실제로 주입하는 것들이다.
- ALFWorld에 넣는 텍스트와 다른 데이터셋에 넣는 텍스트는 글자 단위로 같다.
- ALFWorld 쪽은 과제별 Skill 없이 잰다. 다른 데이터셋에서는 ALFWorld 과제별 Skill을 쓸 수 없어서 짝을 맞춘 것이다.

## 4. 데이터셋

환경과 데이터는 **바꾼 것이 없다.** 전부 라이브러리 기본값이다.

| 데이터셋 | 설정 | 판 수 | 지표 |
|---|---|---|---|
| ALFWorld | 공식 `valid_seen`, 50스텝 | 140 | 성공률 |
| TWC | TextWorldExpress 기본값 (방 3, 물건 4), 100스텝 | test seed 앞 20 (본 실행 100) | 성공률 |
| CookingWorld | TextWorldExpress 기본값 (방 11, 재료 3, 문 있음), 100스텝 | test seed 앞 20 (본 실행 100) | 성공률 |
| ScienceWorld | 단순화 없음, 100스텝, 상태 변화 과제 4개 | 공식 test 변형 36 | 진행 점수 |

수정은 아니지만 논문에 밝혀야 하는 선택:

- TWC·CookingWorld는 원 논문의 게임 파일이 아니라 TextWorldExpress의 재구현이다.
- ScienceWorld는 30개 과제 중 4개(boil, melt, freeze, change-the-state-of-matter-of)만 쓴다. ALFWorld의 데우기·식히기와 절차가 겹친다.
- 프롬프트는 SkillRL의 ALFWorld 형식을 그대로 쓴다(환경 이름만 교체). 이력 2스텝, 온도 0.4.
- ScienceWorld는 가능한 행동이 최대 2,785개라, 두 물체 행동(move·pour·use·dunk·connect)은 든 물건이 첫 대상일 때만 목록에 보여준다. 나머지 환경은 환경이 주는 목록 그대로다.

## 5. 실행

### 설치 (처음 한 번)

```bash
export WORK=/root                      # 연구실 서버면 /dataset/kjs1214
bash scripts/setup_workspace.sh        # 가상환경, 데이터, 모델 다운로드, RL 병합
```

### Step 2 축소판 (RL, seed 1개, 방향 확인용)

```bash
source scripts/env.sh
SEEDS="0" MODELS="rl" SPLITS="seen" BANKS="nosk genonly6 genonly" \
  TRANSFER_ENVS="twc cookingworld scienceworld" TRANSFER_SKILLS="none general6 general" TRANSFER_VARS=20 \
  TWC_PARAMS=default CW_PARAMS=default MAX_STEPS=100 HISTORY=2 \
  OUT=runs_s2mini_default bash scripts/run_general.sh
```

### Step 2 본 실행 (RL + calib, seed 3, 100변형)

위 명령에서 `SEEDS="0 1 2" MODELS="rl calib" TRANSFER_VARS=100 OUT=runs_step2`로 바꾼다.

끝난 조건은 건너뛰므로, 중간에 끊겨도 같은 명령을 다시 걸면 이어서 돈다.

### 결과 보기

```bash
python analysis/general_report.py runs_s2mini_default
python analysis/failure_modes.py "runs_s2mini_default/transfer/*.jsonl"
```

| 보고서 섹션 | 내용 |
|---|---|
| (1a) | ALFWorld, 과제별 Skill 있음: 일반 지침을 뺐을 때의 효과 (Step 1) |
| (1b) | ALFWorld, 과제별 Skill 없음: 일반 지침을 넣었을 때의 효과 |
| (2) | 다른 데이터셋: 일반 지침을 넣었을 때의 효과 |
| (3) | 조건 ①②에 대한 판정 (`MET` / `NOT met` / 바닥) |
| (4) | 짝지은 비교: 각 데이터셋을 절차가 겹치는 ALFWorld 유형 옆에 놓는다 (추가 로그) |
| (5) | ScienceWorld 과제별 결과 (추가 로그) |

(4)의 짝은 결과를 보기 전에 정했다: TWC ↔ pick·pick2, ScienceWorld ↔ heat·cool, CookingWorld ↔ pick·heat(가장 약한 짝).
(4)와 (5)는 기록용이며 실행 내용과 (3)의 판정을 바꾸지 않는다.

`failure_modes.py`는 에피소드가 어떻게 끝났는지(성공 / 환경이 일찍 종료 / 스텝 소진), 목록 안 행동 비율, 같은 행동 반복, 가장 잦은 행동을 찍는다.

## 6. 하네스 검증 (GPU 불필요)

환경이 제공하는 정답 경로를 우리 실험 코드로 재생해서, 우리 설정이 성공을 막지 않는지 확인한다.

```bash
python analysis/gold_check.py twc default     # 정답 20/20, 정답 행동이 전부 목록에 있음
python analysis/gold_check.py cw default      # 정답 20/20, 정답 행동이 전부 목록에 있음
python analysis/gold_check.py sw              # 목록의 행동만으로 정답과 같은 점수, 목록 규칙이 막은 스텝 0
```

ScienceWorld는 100스텝 안에 정답 경로도 다 끝나지 않는 변형이 있다(정답 평균 80~91점). 그래서 진행 점수로 판정한다.

## 7. 폴더 구조

```
scripts/
  setup_workspace.sh      설치
  env.sh                  경로 변수, 가상환경
  run_general.sh          Step 1·2 실행기
  make_skill_variants.py  Skill bank 변형 생성
  merge_fsdp_ckpt.py      RL 체크포인트 병합
alfworld_eval/            ALFWorld 실행 (run_alfworld_eval.py), 게임 목록, 표면 변형, 집계
envs/adapters.py          TWC·CookingWorld·ScienceWorld 어댑터
run_eval.py               다른 데이터셋 실행
skills/sources.py         Skill 텍스트 소스 (none / general6 / general 등)
configs/                  Skill bank 변형 (genonly, genonly6, nogeneral, nogeneral6, mismatched)
analysis/
  general_report.py       보고서
  failure_modes.py        실패 진단
  gold_check.py           하네스 검증
common/                   프롬프트, vLLM 래퍼, SkillRL 메모리 로더
train/                    SkillRL 이어 학습 스크립트 (연구실 GPU용)
```

## 8. 환경변수 (`scripts/run_general.sh`)

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `SEEDS` | `0 1 2` | 반복 seed |
| `MODELS` | `rl calib` | `rl`, `calib`, `base` |
| `SPLITS` | `seen unseen` | ALFWorld 분할. `""`이면 ALFWorld 생략 |
| `BANKS` | `full nogen` | ALFWorld Skill 조건. 과제별 있음: `full` `nogen` `nogen6` / 과제별 없음: `nosk` `genonly6` `genonly` |
| `TRANSFER_ENVS` | `twc cookingworld scienceworld` | 다른 데이터셋. `""`이면 생략 |
| `TRANSFER_SKILLS` | `none general6 general` | 다른 데이터셋 Skill 조건 |
| `TRANSFER_VARS` | 20 | 환경별 변형 수 (ScienceWorld는 과제당 9개가 상한) |
| `TWC_PARAMS`, `CW_PARAMS` | `default` | `default`(라이브러리 기본값) / `reduced` / `easy` 또는 gameParams 문자열 |
| `MAX_STEPS` | 50 | TWC·CookingWorld 스텝 제한. 라이브러리 기본값은 100 |
| `HISTORY` | 2 | 프롬프트에 넣는 이전 스텝 수 |
| `OUT` | `runs_gen` | 결과 폴더 |

ScienceWorld 전용: `SW_SIMPL`(기본 없음), `SW_MAX_STEPS`(기본 100), `SW_LIST`(기본 `held`). 기본값이 라이브러리 기본 설정이다.

## 9. 주의

- 공개 RL 체크포인트는 학습 중간(step 80/150 추정)일 수 있다.
- 한 번 실행한 성공률은 0.1 이상 흔들린다. 본 실행은 seed 3개로 한다.
- 논문 결과로 쓰지 않는 이전 실행: 1차 실험(표면 변형, 쉬운 설정)과 바닥 확인. 쉬운 설정을 썼거나 ScienceWorld 설정에 문제가 있었다. 그 로그에서 "원본"이라 부른 TWC·CookingWorld 설정은 라이브러리 기본값이 아니었다.
- `reduced`·`easy` 설정, ScienceWorld 단순화, 표면 변형은 코드에 옵션으로만 남아 있다.
- `DESIGN_v2.md`, `MEASUREMENTS.md`, `RUNBOOK.md`는 1차 실험 시점의 문서다. 현재 설계는 이 README가 기준이다.
