# 측정 목록 (pilot_v2) — 무엇을, 어떤 조건에서, 어떻게 계산하나

공통: SkillRL 프롬프트 그대로(단일 user 턴, 이력 2스텝, 첫 스텝엔 Skill 없음, <think>/<action>, 가능한 행동 목록), 온도 0.4,
최대 50스텝, ALFWorld seen 140게임, 반복 REPS회. 모든 비교는 같은 게임(또는 변형)끼리 짝지어 부트스트랩 95% CI.

## 조건 축
- 모델: base(Qwen2.5-7B-Instruct, ALFWorld 미학습) / sft(SkillRL SFT) / rl(SkillRL RL, 병합) / calib(Qwen3-30B-A3B, ALFWorld 미학습)
- 표면 변형: none / using / moveto / fromtake / names(13개 이름 동의어, 관찰·과제문·행동목록·Skill 문구 동일 치환)
  * 변형 환경에서 목록에 없는 행동(옛 표현 포함)은 실행되지 않고 'Nothing happens' (실제로 바뀐 인터페이스처럼)
  * Skill 유형 판별은 원래 과제 문장으로 → 원래 환경과 같은 Skill 주입
- Skill 조건: full(SkillRL bank) / nosk(Skill 없음, 템플릿도 Skill 섹션 없음) / mismatch(과제별 Skill을 다른 유형 것으로; 형식 동일)
  / proc(환경 무관 절차 Skill) / proc_mismatch(무관한 절차 Skill, 같은 개수)

## ALFWorld 지표 (summary_alfworld.txt)
| 코드 | 계산 | 해석 |
|---|---|---|
| A | calib(변형) − calib(원래) | ±0.1 안이면 변형이 공정 |
| B | (학습모델 변형−원래) − (calib 변형−원래) | <0이면 학습 모델만 더 무너짐 |
| B' | 원래 환경에서 푼 게임(성공률≥0.5) 중 변형에서 실패한 비율, 학습모델 − calib | 기준 성공률 차이에 덜 민감한 버전 |
| C | 변형 환경 침입률(목록 밖 행동 중 옛 표현과 일치하는 비율), 학습모델 − base/calib | >0이면 학습 표현을 끌어옴 |
| D | 쌍 (full,nosk) (full,mismatch) (proc,proc_mismatch) (proc,nosk)의 성공률 차이: 원래 | 변형 | 변화량 | 원래 작고 변화량>0이면 "은퇴 근거가 표면에 묶임" |
| D' | D 변화량(학습모델) − D 변화량(calib) | >0이면 학습 특유 |
| E | (rl 변형−원래) − (sft 변형−원래) | 부차: RL 단계에서 커졌나 |
| F | 원래 환경, 과제 유형별 full−nosk, full−mismatch | 차이가 작은 유형 = 은퇴 후보 → necessity의 RETIRE_CATS |

## 절차 공유 환경 지표 (summary_transfer.txt) — ScienceWorld(상태 변화), TWC, CookingWorld
- 성공률(과제 완료), 점수(ScienceWorld 0–100을 0–1로, 음수는 0), ALFWorld식 명령 침입률, 형식 오류율
- Skill 소스: none / procedural / mismatch_proc(같은 개수) / skillrl_raw(ALFWorld 문구 그대로, 유형 강제)
- Skill 효과: 각 소스 − none (모델별)
- 핵심: [procedural − none](rl) − [procedural − none](ref), [procedural − mismatch_proc](rl) − [same](ref), ref ∈ {base, calib}
  >0이면 학습한 모델이 절차 Skill에 더 기댐 = 절차를 내재화하지 못함

## 은퇴 필요성 지표 (summary_necessity.txt) — ALFWorld, SkillRL 임베딩 검색(top-k, 유형 무관)
- Library: 원본 / minus_RETIRE_CATS(은퇴) / plus_dups1·2(중복 추가), k ∈ {3, 6}
- 과제를 "은퇴 유형 과제"와 "그 외 과제"로 나눠서: 검색된 과제별 Skill 자리 중 과제 유형 일치 비율, 은퇴 유형이 차지한 비율,
  중복이 차지한 비율, Skill이 하나도 안 잡힌 비율, 성공률(CI), 평균 프롬프트 토큰

## 알려진 한계 (논문에 명시)
- nosk는 프롬프트 형식도 달라짐 → mismatch, proc_mismatch가 형식 통제 대조군
- base는 ALFWorld 성공률이 바닥 → 성공률 비교는 calib, 침입률 비교는 base·calib 둘 다
- 침입률의 "옛 표현"(go to 등)은 자연스러운 영어이기도 함 → 학습 안 한 모델과의 차이(C)로만 해석
- 공개 RL 체크포인트의 학습 step 불명, 학습 중 추가된 Skill은 공개되지 않음
- necessity는 embedding 검색을 쓰지만 공개 RL 모델은 template 모드로 학습됐을 수 있음
- SkillRL template 라우팅은 "put a hot X" 형태의 heat 과제를 pick_and_place로 보냄(원 코드 동작, 요약표에 표시)
