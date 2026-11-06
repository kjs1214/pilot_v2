# pilot_v2 — 절차의 일반화 · 은퇴의 필요성/위험 (SkillRL 공개 체크포인트, 학습 없음)

설계: **DESIGN_v2.md** · 실행 순서/VESSL: **RUNBOOK.md**

```
common/            프롬프트(SkillRL 동일 뼈대), vLLM/mock 생성, SkillRL SkillsOnlyMemory 로더
alfworld_eval/     ALFWorld: 게임 열거, 표면 변형(명령어·이름), 러너(template/embedding 검색, 절차 Skill 소스), Q1·Q2 집계
skills/            환경 무관 절차 Skill 8개 + 환경/과제별 절차 태그 맵 + Skill 소스(procedural/mismatch_proc/skillrl_raw)
envs/adapters.py   ScienceWorld · TWC · CookingWorld 어댑터(동일 인터페이스, ALFWorld 문법 침입 패턴)
run_eval.py        절차 공유 환경 러너 (base/sft/rl × Skill 소스, 병렬 JVM)
metrics_transfer.py  전이 집계: 모델 사다리, Skill 차이, gap(rl)−gap(base), 침입률
necessity/         임베딩 검색 Library 변형(은퇴/중복) 생성 + 검색 자리 점유 집계
configs/           이름 치환 맵, 엇갈린 Skill bank, (생성) Library 변형
scripts/           setup_workspace.sh · smoke.sh · run_all.sh(surface|skillgap|calib|necessity|transfer) · 병합기 · 다운로드
```

여기서 실제로 돌려 확인한 것: ALFWorld(seen 140게임)·ScienceWorld·TWC·CookingWorld 실환경 실행, 변형 되돌리기, Skill 소스 4종,
임베딩 검색(Qwen3-Embedding-0.6B) 자리 점유 집계, FSDP 병합기(가짜 체크포인트), 전체 run_all 파이프라인(가짜 모델).
확인 못 한 것: 실제 GPU에서 vLLM 모델 로드, 진짜 RL 체크포인트 병합 — 스모크 테스트에서 처음 확인.
