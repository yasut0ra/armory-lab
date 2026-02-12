# armory-lab

固定信頼度 (fixed-confidence, PAC) の Best Arm Identification (BAI) を、
研究実験とデモの両方で扱える Python ツールキットです。

## できること

- 環境
  - `bernoulli`: 各腕の報酬が Bernoulli
  - `weapon_damage`: 各腕が 2点混合ダメージ分布（通常ダメージ `d0` / クリダメ `d1` / クリ確率 `p`）
- objective（最強の定義）
  - `dps`: 期待ダメージ最大（`E[damage]` 最大）
  - `oneshot`: `P(damage >= threshold)` 最大
- アルゴリズム
  - `LUCB`
  - `Successive Elimination`
  - `Track-and-Stop`
  - `Top-Two Thompson Sampling (ttts)`
- 出力
  - 推奨腕、停止時刻、腕ごとのサンプル配分、履歴
- 可視化
  - CI推移、サンプル配分、停止時点

## 1分セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e '.[dev]'
```

## 最短で触る（Web UI）

```bash
python -m armory_lab.web --host 127.0.0.1 --port 7860
```

- アクセス: `http://127.0.0.1:7860`
- `env` / `objective` / `threshold` / `algo` を切り替えて実行
- ステータス表は `Weapon` 列（武器名）を常時表示。`weapon_damage` では `d0/d1/crit p` と合わせて比較可能

## CLI

### 既存（bernoulli）

```bash
python -m armory_lab.run --algo lucb --K 20 --delta 0.05 --means topgap:0.05 --seed 0 --plot
```

### weapon_damage + dps

```bash
python -m armory_lab.run \
  --env weapon_damage \
  --algo lucb \
  --objective dps \
  --pack random \
  --K 12 \
  --delta 0.05 \
  --seed 0 \
  --plot
```

### weapon_damage + oneshot

```bash
python -m armory_lab.run \
  --env weapon_damage \
  --algo lucb \
  --objective oneshot \
  --threshold 100 \
  --pack archetypes \
  --K 12 \
  --delta 0.05 \
  --seed 0 \
  --plot
```

### 実用オプション

- `--trials`: 複数seedで連続実行
- `--seed-step`: 試行ごとのseed増分
- `--output-csv`: 試行ごとの結果CSV
- `--json`: JSON出力
- `--means-list`: bernoulliの腕平均を直接指定
- `--json` には `arm_metadata`（`arm`, `weapon_name`, `d0`, `d1`, `p`, `objective_value`）を含む
- `--output-csv` には `weapon_name` 列（`weapon_damage`時は武器名配列）を含む

## レジーム

### bernoulli (`--means`)

- `random`
- `topgap:x`
- `two-groups`

### weapon_damage (`--pack`)

- `random`
  - `d0 in [50,90]`, `d1 in [90,140]`, `p in [0.05,0.35]`
- `topgap:x`
  - best の期待ダメージが 2位より `x` 高いように構成
- `archetypes`
  - 安定型 / クリ型 / ギャンブル型を混合

## 武器鑑定デモ

```bash
python demo/weapon_appraisal.py --env weapon_damage --algo lucb --objective dps --pack archetypes --seed 0
python demo/weapon_appraisal.py --env weapon_damage --algo lucb --objective oneshot --threshold 100 --pack archetypes --seed 0
```

- LUCB時は leader vs challenger をラウンド表示
- 鑑定ゲージ（`gap = LCB_best - maxUCB_others` 由来）を表示

## 実験スイープ

### dps軸（gap sweep）

```bash
python experiments/sweep.py \
  --env weapon_damage \
  --objective dps \
  --algo lucb \
  --K-values 10,20 \
  --deltas 0.05,0.1 \
  --gaps 4,8,12 \
  --pack-regime topgap \
  --trials 50 \
  --output experiments/sweep_weapon_dps.csv
```

### oneshot軸（threshold sweep）

```bash
python experiments/sweep.py \
  --env weapon_damage \
  --objective oneshot \
  --algo lucb \
  --K-values 10,20 \
  --deltas 0.05,0.1 \
  --thresholds 90,100,110 \
  --pack-regime archetypes \
  --trials 50 \
  --output experiments/sweep_weapon_oneshot.csv
```

CSVには `env/objective/threshold` を含みます。

## 実装メモ

- CIは Hoeffding 型（時刻依存 `delta_{i,t}`）
- `weapon_damage` の `dps` は有界報酬レンジでCI幅をスケール
- `oneshot` は内部的に `hit = 1(damage >= threshold)` として扱い、Bernoulli型で同定

## テスト / 型チェック

```bash
pytest -q
mypy armory_lab
```
