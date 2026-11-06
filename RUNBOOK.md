# 실행 순서 (VESSL, A100 1장 기준)

## 0. 준비물 체크
- [ ] 맥: `gh auth login`, `~/vessl-tools` + `./setup.sh` 완료, `vesslctl auth status`에 연구실 조직이 보임
- [ ] SSH 키: VESSL 콘솔 Settings → SSH Keys → Generate Key로 받은 파일을 `~/.ssh/`에 두고 `chmod 600`
- [ ] 이 코드(`pilot_line12`)를 본인 GitHub 비공개 저장소에 push

```bash
# 맥
cd ~/Downloads && unzip pilot_line12.zip && cd pilot_line12
git init && git add . && git commit -m "pilot line1/line2"
gh repo create pilot_line12 --private --source . --push
```

## 1. 워크스페이스 만들기 (웹 콘솔 또는 `./create_workspace.sh pilot`)
- GPU: A100 80GB 1장
- **Persistent volume: Cluster storage를 `/root`에 마운트** (코드·venv·데이터·모델·결과가 전부 `~` 아래라
  일시정지·종료 후에도 남게). 드롭다운에 볼륨이 없으면 연구실 관리자에게 팀 볼륨을 요청. 200GB 이상 권장.
- 이미지: CUDA 12.8 + Python 3.11 (또는 3.12) 권장. torch 버전은 무관(vLLM이 venv 안에 자기 torch를 설치).
  setup 스크립트가 `~/venv` 가상환경을 만들어 씁니다. 이미지 Python이 3.13이면 3.12를 따로 받아서 만듭니다
  (TextWorld/ALFWorld는 Python 3.13에서 오류).
- 접속: Connect 탭 → JupyterLab → File → New → Terminal (SSH 없이 브라우저로 충분)
- 워크스페이스 이름(slug)을 적어두세요. 아래에서 `<slug>`로 씁니다.

## 2. 설치·데이터·모델 (워크스페이스, 1회, 30~90분)
JupyterLab 파일 목록에 `pilot_line12.zip`을 업로드한 뒤 터미널에서:
```bash
cd ~ && unzip -o pilot_line12.zip && cd pilot_line12      # unzip이 없으면: python3 -m zipfile -e pilot_line12.zip .
bash scripts/setup_workspace.sh     # Python 3.12 venv, vLLM 등 설치, SkillRL, ALFWorld 데이터, 모델 다운로드·RL 병합, Java
```
끝에 출력되는 `cuda available: True`, GPU 이름·메모리(80GB), 디스크 여유를 확인하세요.
이후 실행 스크립트들은 `~/venv`를 자동으로 켭니다. 손으로 파이썬을 칠 때만 `source ~/venv/bin/activate`.

## 3. 스모크 테스트 (5~15분)
```bash
bash scripts/smoke.sh          # 과제 유형별 1개씩 6게임, RL 모델
```
마지막 줄의 걸린 시간을 알려주면 본 실행 시간을 다시 계산할 수 있습니다.

## 4. 자동 멈춤 시험 (2분)
```bash
nohup bash -c "sleep 60; vesslctl workspace pause <slug>" > /tmp/pause_test.log 2>&1 &
```
1분 뒤 워크스페이스가 멈추면 성공. 다시 켜기: 맥에서 `vesslctl workspace start <slug>` 후 재접속.

## 5. 본 실행 (끝나면 자동으로 pause)
```bash
cd ~/pilot_line12
WORKSPACE_SLUG=<slug> nohup bash scripts/run_all.sh all > runs/run_all.log 2>&1 &
tail -f runs/run_all.log        # 진행 확인 (Ctrl+C로 보기만 종료, 실행은 계속)
```
- 단계별로 나눠 돌리려면 `all` 대신:
  - `line1` : rl/sft/base × {원래, using, moveto, fromtake, names}, 진짜 Skill bank
  - `skills`: sft/rl × {원래, moveto, names} × {Skill 없음, 엇갈린 과제별 Skill} ← Skill 의존도 조건
  - `calib` : Qwen3-30B-A3B 보정 모델 × 같은 변형 (80GB 필요)
  - `line2` : ScienceWorld 상태 변화 과제, base/sft/rl
- 추천 순서: `line1` → `skills` → `calib` → `line2`. 각각 따로 걸면 중간에 결과를 보면서 갈 수 있어요.
- 이미 끝난 조건은 건너뛰므로, 중간에 끊겨도 같은 명령을 다시 실행하면 이어서 돕니다.
- 반복 횟수는 `REPS=2`(기본), ScienceWorld 변형 수는 `SW_VARS=10`(기본)로 조절.

## 6. 결과 확인
```bash
cat runs/summary_alfworld.txt      # Q1·Q2 (ALFWorld)
cat runs/summary_transfer.txt      # 절차 공유 환경: 모델 사다리, Skill 차이, 침입률
cat runs/summary_necessity.txt     # 임베딩 검색에서 은퇴 대상/중복 Skill이 차지한 자리
```
summary_alfworld.txt 맨 아래 **KEY CONTRASTS**(ALFWorld), summary_transfer.txt(절차 공유 환경), summary_necessity.txt(은퇴 필요성) 세 파일이 결과예요(모두 같은 게임끼리 짝지은 차이, 95% CI).

**Q1. 학습 안 한 모델은 그대로인데, 학습한 모델만 바꾼 쪽에서 무너지는가**
- [A] calib(변형) − calib(원래)가 ±margin 안이면 FAIR (바뀐 표현이 원래 더 어려운 게 아님)
- [B] (학습 모델 변형−원래) − (calib 변형−원래) < 0 이면 TRAINED BREAKS
- [C] 변형 환경 침입률, 학습 모델 − 학습 안 한 모델(base, calib) > 0 이면 TRAINED INTRUDES

**Q2 (메인). 원래 환경에서는 Skill 있이/없이 차이가 작은 학습 모델이, 바뀐 환경에서는 그 차이가 커지는가**
- [D] Skill 차이 = 성공률(진짜 Skill) − 성공률(Skill 없음 또는 엇갈린 Skill). 원래 환경 차이 | 변형 환경 차이 | 변화량.
  rl에서 "원래 차이는 작고, 변화량 > 0"(SKILLS MATTER MORE AFTER CHANGE)이면 가설대로.
- [D'] 대조: 그 변화량이 학습 안 한 calib보다 큰가 (TRAINING-SPECIFIC). calib도 똑같이 커지면
  "표현이 바뀌면 누구에게나 Skill이 더 도움 된다"일 뿐이라 학습 특유의 현상이 아님.
- 엇갈린 Skill(mismatch)은 "Skill 텍스트가 있다는 것" 자체가 아니라 "맞는 내용"이 중요한지 확인하는 형식 대조군.
- 은퇴와의 연결: 원래 환경의 작은 Skill 차이 = 은퇴 규칙이 "필요 없다"고 보는 근거. 그 차이가 바뀐 환경에서 커지면,
  학습 분포에서 내린 은퇴 판단이 확장 상황에서 틀린다는 직접 증거.

**부차**: [E] 무너짐이 RL 단계에서 커졌는지 (rl − sft)

검정력: 게임 140개 × 2회. 같은 게임끼리 결과가 비슷하면 CI가 ±0.07~0.1 정도(가짜 데이터로 확인). REPS=1은 방향 확인용.

## 시간·크레딧 (A100 80GB 1장, 1.55 크레딧/시간, 예상치)
| 단계 | 에피소드 | 시간 | 크레딧 |
|---|---|---|---|
| 설치·다운로드·병합 | | 1~1.5h | 2 |
| 스모크·멈춤 시험 | 6 | 0.3h | 0.5 |
| surface (rl·sft·base × 5환경 × 140 × 2) | 4,200 | 5~9h | 8~14 |
| skillgap (Skill 없음 5환경, 엇갈린 3환경, 절차 2×2환경, rl·sft) | 5,600 | 6~12h | 9~19 |
| calib (full 5환경, 없음 3환경, 절차 2환경; 30B) | 2,800 | 4~7h | 6~11 |
| necessity (rl, k 3·6 × Library 4종 × 140 × 2) | 2,240 | 3~5h | 5~8 |
| transfer (3환경 × 3모델 × 4소스 × 10변형 × 2회, 8개 동시) | 720 | 2~4h | 3~6 |
| 합계 | | 21~39h | 34~61 |

- 근거: ALFWorld 환경 한 스텝 약 0.1초(측정) + 생성 시간 → 에피소드당 4~8초. 스모크 테스트 시간으로 다시 계산.
- 첫 판(약 10~16 크레딧): 반복 1회, 핵심만
  ```bash
  export REPS=1 SURFACE_MODELS="rl sft" MISMATCH_PERTS="none names" TRANSFER_VARS=6
  nohup bash -c "bash scripts/run_all.sh surface; bash scripts/run_all.sh skillgap; WORKSPACE_SLUG=<slug> bash scripts/run_all.sh transfer" > runs/pass1.log 2>&1 &
  ```
  신호가 보이면 `REPS=2`, base, calib, necessity를 추가로 돌리면 돼요(끝난 조건은 건너뜀).
- 켜두고 노는 시간이 제일 큰 낭비예요. 본 실행은 꼭 `WORKSPACE_SLUG=<slug>`를 붙여 끝나면 자동 pause.

## 막히면
- `CUDA out of memory`: `--gpu_mem 0.8`, 또는 `--batch 16`으로 동시 게임 수 축소
- 보정 모델이 안 올라감: A100이 40GB이면 calib만 `--preset h100` 워크스페이스에서 `bash scripts/run_all.sh calib`
- Hugging Face 다운로드 제한: `export HF_TOKEN=<Read 토큰>` 후 다시 실행
- ScienceWorld가 Java 없음: `apt-get install -y default-jre-headless` 또는 `conda install -c conda-forge openjdk`
- ScienceWorld 점수가 전부 0 근처: run_eval.py에 `--simplifications noElectricalAction,openDoors`(모든 모델 동일하게)
- TWC/CookingWorld가 너무 쉽거나 어려우면 `--game_params`로 조절(기본: 방 3개, 치울 물건 2개 / 재료 2개)
- GPU 없이 설치만 점검: `PILOT_MOCK_LLM=1 MAX_GAMES=3 TRANSFER_VARS=1 REPS=1 bash scripts/run_all.sh all`
  (모델 대신 무작위 행동. CPU 워크스페이스에서 파이프라인 점검용)

## 알아둘 점
- SkillRL template 모드는 과제 문장의 키워드로 Skill 유형을 고릅니다. "put a hot X in Y" 형태의 heat 과제는
  'heat'라는 단어가 없어서 pick_and_place Skill이 들어갑니다. 공개 코드 그대로의 동작이라 그대로 재현하며,
  `summary_alfworld.txt`에 "skill routing ... differs" 줄로 표시됩니다.
- 변형 환경에서도 Skill 유형 판별은 원래 과제 문장으로 하므로, 원래 환경과 같은 Skill이 들어갑니다.
