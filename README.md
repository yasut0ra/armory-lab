# armory-lab

研究寄りの Best Arm Identification (BAI) ツールキット + 見て楽しい武器鑑定デモです。  
Python で fixed-confidence (PAC) 設定の最良腕同定を最小構成で実装しています。

## 実装済みMVP

- 環境: Bernoulli bandit (`mu` ごとに Bernoulli 報酬)
- fixed-confidence BAI アルゴリズム:
  - `Successive Elimination` (Action Elimination)
  - `LUCB`
- 信頼区間: Hoeffding 型 + 時刻依存の信頼割当
  - `delta_{i,t} = delta / (2 K t^2)` を使い、`sum_{i,t} delta_{i,t} <= delta` を満たす
- メトリクス:
  - 誤識別率
  - 停止時刻（総サンプル数）
  - 各腕のサンプル配分
- 可視化 (`matplotlib`):
  - CI推移
  - サンプル配分
  - 停止時点

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e .[dev]
```

## CLI 実行

```bash
python -m armory_lab.run --algo lucb --K 20 --delta 0.05 --means topgap:0.05 --seed 0 --plot
```

`--means` は以下をサポート:

- `random`: `mu ~ Uniform(0,1)`
- `topgap:x`: `best=0.5+x`, `second=0.5`, その他はその下
- `two-groups`: 上位群/下位群の2群で `mu` を生成

plot 保存例:

```bash
python -m armory_lab.run --algo se --K 15 --means two-groups --plot --save-plot artifacts/se_two_groups.png --no-show
```

## デモ（武器鑑定）

```bash
python demo/weapon_appraisal.py --algo lucb --delta 0.05 --seed 0 --plot
```

ラウンドごとに以下を表示します:

- 試した武器
- 推定値と信頼区間
- 候補武器の絞り込み

最後に「最強武器を鑑定完了（delta=...）」と試行回数、サンプル配分を表示します。

## 実験スイープ

```bash
python experiments/sweep.py \
  --algo lucb \
  --K-values 10,20 \
  --deltas 0.05,0.1 \
  --gaps 0.03,0.05,0.1 \
  --trials 50 \
  --output experiments/sweep_results.csv
```

CSV 出力:

- `mean_total_pulls`
- `misidentification_rate`
- `mean_pulls_per_arm`

## アルゴリズム概要

- `Successive Elimination`: 残存腕をラウンドごとに引き、CIに基づいて劣る腕を除去
- `LUCB`: 現在の最良推定腕とその最有力対抗腕を重点的に引き、`LCB(best) >= UCB(others)` で停止

どちらも fixed-confidence で停止し、`recommend_arm`, `total_pulls`, `pulls_per_arm`, `history` を返します。
