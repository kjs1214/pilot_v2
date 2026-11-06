# 설계 v2 — 절차의 일반화와 은퇴의 필요성·위험

## 연구 질문
Skill 학습(내재화+외재화)은 Skill에 담긴 **절차**(전제 조건과 순서: 찾기 → 집기 → 기구로 가져가 상태 바꾸기 → 목적지에 놓기)를
익히는가, 아니면 학습 환경의 **표면**(명령어 표현, 물건 이름)에 묶어 익히는가. 그 절차는 같은 절차를 공유하는 다른 환경으로 옮겨가는가.
은퇴는 이 답에 따라 정당하거나 위험해지는 결정이며, 은퇴의 필요성은 검색이 해결하지 못하는 비용으로 따로 보인다.

## 세 축과 그에 맞는 데이터셋
| 축 | 질문 | 환경 | 바꾸는 것 |
|---|---|---|---|
| 표면 | 표현만 바꿔도 학습 모델만 무너지나 | ALFWorld (seen) | 명령어 표현(AgentRefine·AgentRM 규칙), 물건 이름 동의어 |
| 절차 공유 | 같은 절차를 가진 다른 환경에서 절차 Skill이 통하나 | ScienceWorld(상태 변화: boil/melt/freeze/change), TWC(찾아서 제자리에), CookingWorld(지시 읽고 재료 찾아 조리) | 환경·명령어·물건 전부 (절차 구조만 공유) |
| 은퇴 필요성 | 내재화된 Skill이 검색 자리를 차지하나 | ALFWorld, 임베딩 검색(top-k, 유형 무관) | Library 구성(은퇴/중복 추가), k |

데이터셋은 억지로 맞추지 않고 절차 구조가 겹치는 것만 골랐다. 왜 겹치는지는 `skills/procedure_map.json`의 태그가 명시한다.

## Skill 소스 (교차 환경)
- `none`: Skill 없음
- `procedural`: 환경 이름이 전혀 없는 절차 Skill 8개(`skills/procedural_skills.json`, SkillRL ALFWorld bank에서 손으로 추상화).
  환경·과제의 절차 태그로 선택(검색 없음).
- `mismatch_proc`: 해당 과제와 무관한 절차 Skill(형식 동일, 내용 틀림) — 형식 대조군
- `skillrl_raw`: SkillRL의 ALFWorld Skill을 그대로(전자레인지 등 ALFWorld 이름 포함) 넣음 — 표면에 묶인 Skill이 다른 환경에서 도움이 되는지/해가 되는지

## 단계와 핵심 비교
1. **surface / skillgap / calib (ALFWorld)** — summary_alfworld.txt의 KEY CONTRASTS
   - Q1: [A] 학습 안 한 모델은 그대로, [B] 학습 모델만 무너짐, [C] 학습 표현 침입
   - Q2: [D] 원래 환경에선 작던 Skill 차이가 표현을 바꾸면 커짐 (`nosk`, `mismatch`, `proc`, `proc_mismatch` 각각), [D'] 보정 모델 대비
2. **transfer (ScienceWorld / TWC / CookingWorld)** — summary_transfer.txt
   - 모델 사다리 base → sft → rl: ALFWorld 학습이 절차 공유 환경에서의 성능을 올리는가, 내리는가
   - Skill 차이: `procedural` − `none` (절차 Skill이 통하나), `skillrl_raw` − `none` (표면 묶인 Skill은 통하나/해가 되나), `mismatch_proc` (형식 대조)
   - 핵심: gap(rl) − gap(base) — 학습한 모델이 절차 Skill에 **더** 기대면, 학습이 절차를 내재화하지 못하고 외부 Skill을 필요로 한다는 뜻
   - ALFWorld식 명령 침입률: 학습 모델이 다른 환경에서 ALFWorld 문법을 끌어오는가
3. **necessity (ALFWorld, 임베딩 검색)** — summary_necessity.txt
   - Library 변형: full / minus(RETIRE_CATS: Q2에서 Skill 차이가 작았던 유형) / +중복 1·2배, k ∈ {3, 6}
   - 지표: 검색된 과제별 Skill 자리 중 (a) 과제 유형과 맞는 비율, (b) 은퇴 대상 유형이 차지한 비율, (c) 중복이 차지한 비율, 성공률, 프롬프트 토큰
   - 주장: 내재화된 Skill은 과제와 가장 비슷해서 검색 자리를 차지하지만 정책에는 쓸모가 없다 → 검색이 못 하는 정책 기준 덜어내기(은퇴)가 필요

## 판정 (가설대로면)
- Q1: A=FAIR, B=TRAINED BREAKS, C=TRAINED INTRUDES
- Q2: rl에서 none 차이 작음 + 변형에서 SKILLS MATTER MORE AFTER CHANGE, 보정 모델 대비 TRAINING-SPECIFIC
- transfer: procedural HELPS (특히 rl), skillrl_raw는 no effect 또는 HURTS, rl 침입률 > base
- necessity: minus 변형에서 과제 유형 일치 비율↑·성공률 유지 또는 ↑, +중복에서 중복 점유율↑·성공률↓

반대로 나오면: 절차를 제대로 익혔다(은퇴 안전) / 절차 Skill이 환경 간에 안 통한다(절차 추상화 자체가 어렵다)는 결과이고, 그것도 보고한다.
