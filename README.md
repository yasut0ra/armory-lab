# armory-lab

固定信頼度 (fixed-confidence, PAC) の Best Arm Identification (BAI) を、
「研究コードとして読みやすく」「デモとして触って楽しい」を両立する最小構成で実装したリポジトリです。

## このリポジトリでできること

- Bernoulli bandit 環境で最良腕を同定
- アルゴリズム: `LUCB` / `Successive Elimination`
- 指標: 誤識別率、停止時刻（総試行数）、腕ごとのサンプル配分
- 可視化: CI推移、サンプル配分、停止時点
- 実行手段: CLI / Web UI / 武器鑑定デモ / 実験スイープ

## 1分セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e '.[dev]'
```

## 最初に触るなら Web UI

```bash
python -m armory_lab.web --host 127.0.0.1 --port 7860
```

- アクセス先: `http://127.0.0.1:7860`
- 画面左: パラメータ入力
- 画面右: 推奨腕、停止理由、CI推移、サンプル配分

### Web UI でよく使う入力例

- `means=random`: 難易度が毎回変わる
- `means=topgap:0.05`: 1位と2位が近く、やや難しい
- `means=topgap:0.10`: ギャップが大きく、停止が早い
- `means=two-groups`: 上位群と下位群に分かれる

### 結果の見方

- `推奨腕`: アルゴリズムが最良と判断した腕
- `真の最良腕`: 生成した真の平均での最良腕
- `停止時の総試行数`: 判定までに必要だったサンプル数
- `CI推移`: 推定平均と信頼区間の時間推移
- `サンプル配分`: どの腕を重点的に引いたか

## CLI で実行

```bash
python -m armory_lab.run --algo lucb --K 20 --delta 0.05 --means topgap:0.05 --seed 0 --plot
```

画像保存だけしたい場合:

```bash
python -m armory_lab.run --algo se --K 15 --means two-groups --plot --save-plot artifacts/se_two_groups.png --no-show
```

## 武器鑑定デモ (ログ重視)

```bash
python demo/weapon_appraisal.py --algo lucb --delta 0.05 --seed 0 --plot
```

各ラウンドで以下を表示します。

- どの武器を試したか
- 推定値と信頼区間
- 候補がどう絞られていくか

## 実験スイープ (CSV出力)

```bash
python experiments/sweep.py \
  --algo lucb \
  --K-values 10,20 \
  --deltas 0.05,0.1 \
  --gaps 0.03,0.05,0.1 \
  --trials 50 \
  --output experiments/sweep_results.csv
```

CSV列:

- `mean_total_pulls`
- `misidentification_rate`
- `mean_pulls_per_arm`

## アルゴリズム概要

- `Successive Elimination`
  - 残っている腕をラウンドで順に引く
  - CIで明確に劣る腕を除去
  - 1本になったら停止
- `LUCB`
  - 現在の最良推定腕と最有力対抗腕を重点サンプリング
  - `LCB(best) >= UCB(challenger)` で停止

どちらも fixed-confidence 設定で停止し、次を返します。

- `recommend_arm`
- `total_pulls`
- `pulls_per_arm`
- `history`

## 信頼区間の設計

Hoeffding型の信頼半径を使い、
時刻依存の割当として `delta_{i,t} = delta / (2 K t^2)` を採用しています。
この設計で `sum_{i,t} delta_{i,t} <= delta` を満たし、全時刻同時保証の形にしています。

## テストと型チェック

```bash
pytest -q
mypy armory_lab
```
