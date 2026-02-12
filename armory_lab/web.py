from __future__ import annotations

import argparse
import base64
import io
from dataclasses import dataclass

import matplotlib
import numpy as np
from flask import Flask, Response, render_template_string, request

from armory_lab.algos.base import HistoryRecord
from armory_lab.run import RunConfig, build_algo, build_problem

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(slots=True)
class RoundPreview:
    round_id: int
    total_pulls: int
    a_hat: int | None
    selected_arms: str
    active_count: int


@dataclass(slots=True)
class WebRunResult:
    config: dict[str, str]
    recommend_arm: int
    true_best: int
    total_pulls: int
    pulls_per_arm: list[int]
    ci_plot_b64: str
    alloc_plot_b64: str
    rounds_preview: list[RoundPreview]
    stop_reason: str
    top_true_arms: list[tuple[int, float]]
    metric_label: str
    status_headers: list[str]
    status_rows: list[list[str]]


DEFAULT_FORM: dict[str, str] = {
    "env": "bernoulli",
    "algo": "lucb",
    "objective": "dps",
    "threshold": "",
    "k": "20",
    "delta": "0.05",
    "means": "topgap:0.05",
    "pack": "random",
    "seed": "0",
    "max_pulls": "200000",
}


HTML_TEMPLATE = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Armory Lab | BAI Console</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #f6f4ef;
      --ink: #161616;
      --card: #fffdf8;
      --muted: #5b5651;
      --line: #d5d0c7;
      --brand-a: #c8553d;
      --brand-b: #1d4e89;
      --ok: #157f3a;
      --bad: #b22222;
      --soft: #f8f4ed;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Space Grotesk", sans-serif;
      background:
        radial-gradient(circle at 7% 10%, #f3cc9e6f 0, transparent 35%),
        radial-gradient(circle at 86% 86%, #a8c9ee75 0, transparent 30%),
        linear-gradient(155deg, #f8f4eb 0%, #f0eadf 42%, #e8eff8 100%);
      min-height: 100vh;
    }
    .wrap {
      max-width: 1240px;
      margin: 0 auto;
      padding: 24px 20px 48px;
      animation: rise .38s ease-out;
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }
    h1 {
      margin: 0 0 8px;
      font-size: clamp(1.75rem, 3.2vw, 2.4rem);
      letter-spacing: .01em;
    }
    .lead { margin: 0; color: var(--muted); line-height: 1.5; }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 10px 26px #0000000e;
      padding: 16px;
    }
    .intro { margin-bottom: 16px; }
    .steps {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .step {
      background: var(--soft);
      border: 1px solid #e8e0d3;
      border-radius: 12px;
      padding: 10px;
      font-size: .87rem;
      line-height: 1.45;
    }
    .step b { display: block; margin-bottom: 4px; color: #2f2a24; }
    .layout {
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 16px;
      align-items: start;
    }
    .stack { display: grid; gap: 12px; }
    label {
      display: block;
      font-size: .9rem;
      color: #5c574f;
      margin-bottom: 4px;
    }
    .hint {
      display: block;
      margin-top: 4px;
      color: #7a736b;
      font-size: .78rem;
      line-height: 1.4;
    }
    input, select {
      width: 100%;
      border: 1px solid #cfc9be;
      border-radius: 10px;
      background: #fff;
      color: #1e1e1e;
      font-size: .95rem;
      padding: 9px 10px;
      outline: none;
    }
    input:focus, select:focus {
      border-color: var(--brand-b);
      box-shadow: 0 0 0 3px #1d4e8920;
    }
    .btn {
      width: 100%;
      border: 0;
      border-radius: 12px;
      padding: 11px 12px;
      background: linear-gradient(105deg, var(--brand-a), #df6f53);
      color: #fff;
      font-size: 0.99rem;
      font-weight: 700;
      cursor: pointer;
      transition: transform .12s ease, filter .12s ease;
    }
    .btn:hover { transform: translateY(-1px); filter: saturate(1.07); }
    .mono { font-family: "JetBrains Mono", monospace; }
    .error {
      border-left: 4px solid var(--bad);
      color: #5b1515;
      background: #fff3f3;
      padding: 10px;
      border-radius: 8px;
      font-size: .9rem;
    }
    .metric-row {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    .metric {
      border-left: 4px solid var(--brand-b);
      border-radius: 10px;
      background: #faf8f3;
      padding: 10px 10px 10px 12px;
    }
    .metric .k { color: #645f58; font-size: .78rem; margin-bottom: 4px; }
    .metric .v { font-size: 1.08rem; font-weight: 700; }
    .ok { color: var(--ok); }
    .bad { color: var(--bad); }
    .sub-grid {
      display: grid;
      grid-template-columns: 1.3fr .9fr;
      gap: 12px;
      margin-bottom: 12px;
    }
    .box {
      border: 1px solid #ded6ca;
      background: #fff;
      border-radius: 10px;
      padding: 10px;
      font-size: .88rem;
      line-height: 1.45;
      color: #3c3a36;
    }
    .box h3 { margin: 0 0 6px; font-size: .95rem; }
    .charts { display: grid; gap: 12px; }
    figure { margin: 0; }
    figure img {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
    }
    figcaption { color: #635d55; font-size: .8rem; margin-top: 4px; }
    .table-wrap {
      margin-top: 10px;
      max-height: 280px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
    }
    table { width: 100%; border-collapse: collapse; font-size: .84rem; }
    th {
      text-align: left;
      position: sticky;
      top: 0;
      background: #faf7f1;
      border-bottom: 1px solid var(--line);
      padding: 8px;
    }
    td {
      border-bottom: 1px solid #f1ece4;
      padding: 8px;
      vertical-align: top;
    }
    .empty { color: #605b53; line-height: 1.55; font-size: .92rem; }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .steps { grid-template-columns: 1fr; }
      .metric-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .sub-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="wrap">
    <section class="card intro">
      <h1>Armory Lab BAI Console</h1>
      <p class="lead">env と objective を切り替えて、最強武器の定義を比較できます（DPS最強 / ワンパン最強）。</p>
      <div class="steps">
        <div class="step"><b>Step 1: 環境を選ぶ</b>Bernoulli または weapon_damage を選びます。</div>
        <div class="step"><b>Step 2: 目的を選ぶ</b>DPS最大か、閾値到達確率最大かを選びます。</div>
        <div class="step"><b>Step 3: 結果を読む</b>推奨腕、停止時刻、CI推移、サンプル配分を確認します。</div>
      </div>
    </section>

    <section class="layout">
      <form class="card stack" method="post">
        <div>
          <label for="env">環境</label>
          <select id="env" name="env">
            <option value="bernoulli" {% if form.env == "bernoulli" %}selected{% endif %}>bernoulli</option>
            <option value="weapon_damage" {% if form.env == "weapon_damage" %}selected{% endif %}>weapon_damage</option>
          </select>
        </div>

        <div>
          <label for="algo">アルゴリズム</label>
          <select id="algo" name="algo">
            <option value="lucb" {% if form.algo == "lucb" %}selected{% endif %}>LUCB</option>
            <option value="se" {% if form.algo == "se" %}selected{% endif %}>Successive Elimination</option>
            <option value="tas" {% if form.algo == "tas" %}selected{% endif %}>Track-and-Stop</option>
          </select>
        </div>

        <div>
          <label for="objective">objective</label>
          <select id="objective" name="objective">
            <option value="dps" {% if form.objective == "dps" %}selected{% endif %}>dps (期待ダメージ)</option>
            <option value="oneshot" {% if form.objective == "oneshot" %}selected{% endif %}>oneshot (到達確率)</option>
          </select>
        </div>

        <div>
          <label for="threshold">threshold (oneshot時に必須)</label>
          <input id="threshold" name="threshold" type="number" step="0.1" value="{{ form.threshold }}" />
        </div>

        <div>
          <label for="k">K (腕の本数)</label>
          <input id="k" name="k" type="number" min="2" max="80" value="{{ form.k }}" />
        </div>

        <div>
          <label for="delta">delta</label>
          <input id="delta" name="delta" type="number" step="0.001" min="0.001" max="0.999" value="{{ form.delta }}" />
        </div>

        <div>
          <label for="means">bernoulli means regime</label>
          <input id="means" name="means" type="text" value="{{ form.means }}" />
          <span class="hint">例: random / topgap:0.05 / two-groups</span>
        </div>

        <div>
          <label for="pack">weapon pack regime</label>
          <input id="pack" name="pack" type="text" value="{{ form.pack }}" />
          <span class="hint">例: random / topgap:8 / archetypes</span>
        </div>

        <div>
          <label for="seed">seed</label>
          <input id="seed" name="seed" type="number" value="{{ form.seed }}" />
        </div>

        <div>
          <label for="max_pulls">max pulls</label>
          <input id="max_pulls" name="max_pulls" type="number" min="1000" max="1000000" value="{{ form.max_pulls }}" />
        </div>

        <button class="btn" type="submit">Run Appraisal</button>
      </form>

      <section class="card">
        {% if error %}
          <div class="error mono">{{ error }}</div>
        {% elif out %}
          <div class="metric-row">
            <article class="metric">
              <div class="k">推奨腕</div>
              <div class="v mono">{{ out.recommend_arm }}</div>
            </article>
            <article class="metric">
              <div class="k">真の最良腕</div>
              <div class="v mono">{{ out.true_best }}</div>
            </article>
            <article class="metric">
              <div class="k">判定一致</div>
              <div class="v {% if out.recommend_arm == out.true_best %}ok{% else %}bad{% endif %}">
                {% if out.recommend_arm == out.true_best %}YES{% else %}NO{% endif %}
              </div>
            </article>
            <article class="metric">
              <div class="k">停止時の総試行数</div>
              <div class="v mono">{{ out.total_pulls }}</div>
            </article>
          </div>

          <div class="sub-grid">
            <section class="box">
              <h3>実行サマリ</h3>
              <div class="mono">
                env={{ out.config["env"] }} / objective={{ out.config["objective"] }} / threshold={{ out.config["threshold"] if out.config["threshold"] else "-" }}<br>
                algo={{ out.config["algo"] }} / K={{ out.config["k"] }} / delta={{ out.config["delta"] }} / seed={{ out.config["seed"] }}
              </div>
              <div style="margin-top:6px;"><b>停止理由:</b> {{ out.stop_reason }}</div>
            </section>
            <section class="box">
              <h3>真の objective 値 上位5腕</h3>
              <div class="mono">
                {% for arm_idx, mu in out.top_true_arms %}
                  arm {{ arm_idx }}: {{ "%.4f"|format(mu) }}<br>
                {% endfor %}
              </div>
            </section>
          </div>

          <h3 style="margin:14px 0 4px;font-size:1rem">武器/腕ステータス比較</h3>
          <div style="font-size:.82rem;color:#6a655e;margin-bottom:6px;">
            停止時点の推定 objective 値が高い順です（上ほど有力）。記号: ★=真の最良, ◎=推奨腕, ★◎=一致。
          </div>
          {% if out.config["env"] == "weapon_damage" %}
            <div style="font-size:.82rem;color:#6a655e;margin-bottom:8px;line-height:1.45;">
              weapon_damage の列定義: d0=通常ダメージ, d1=クリティカルダメージ, crit p=クリティカル確率。<br>
              1回の攻撃は確率 (1-p) で d0、確率 p で d1。E[dmg] = d0*(1-p) + d1*p
            </div>
          {% endif %}
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  {% for header in out.status_headers %}
                    <th>{{ header }}</th>
                  {% endfor %}
                </tr>
              </thead>
              <tbody class="mono">
                {% for row in out.status_rows %}
                  <tr>
                    {% for cell in row %}
                      <td>{{ cell }}</td>
                    {% endfor %}
                  </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>

          <div class="charts">
            <figure>
              <img src="data:image/png;base64,{{ out.ci_plot_b64 }}" alt="CI trajectory" />
              <figcaption>{{ out.metric_label }} の推移と信頼区間（停止時点まで）。</figcaption>
            </figure>
            <figure>
              <img src="data:image/png;base64,{{ out.alloc_plot_b64 }}" alt="Allocation" />
              <figcaption>サンプル配分（どの腕を重点的に引いたか）。</figcaption>
            </figure>
          </div>

          <h3 style="margin:14px 0 8px;font-size:1rem">直近ラウンド</h3>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>round</th>
                  <th>pulls</th>
                  <th>a_hat</th>
                  <th>selected</th>
                  <th>active_count</th>
                </tr>
              </thead>
              <tbody class="mono">
                {% for row in out.rounds_preview %}
                  <tr>
                    <td>{{ row.round_id }}</td>
                    <td>{{ row.total_pulls }}</td>
                    <td>{{ row.a_hat }}</td>
                    <td>{{ row.selected_arms }}</td>
                    <td>{{ row.active_count }}</td>
                  </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        {% else %}
          <div class="empty">
            左のフォームを設定して <span class="mono">Run Appraisal</span> を押してください。<br>
            例: <span class="mono">env=weapon_damage / objective=oneshot / threshold=100</span>
          </div>
        {% endif %}
      </section>
    </section>
  </main>
</body>
</html>
"""


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_ci_plot(history: list[HistoryRecord], n_arms: int, metric_label: str) -> str:
    x = np.asarray([item.total_pulls for item in history], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = plt.get_cmap("tab20")
    for arm in range(n_arms):
        means = np.asarray([item.means[arm] for item in history], dtype=np.float64)
        lcbs = np.asarray([item.lcbs[arm] for item in history], dtype=np.float64)
        ucbs = np.asarray([item.ucbs[arm] for item in history], dtype=np.float64)
        color = cmap(arm % 20)
        ax.plot(x, means, color=color, linewidth=1.1)
        ax.fill_between(x, lcbs, ucbs, color=color, alpha=0.08)

    ax.axvline(float(x[-1]), linestyle="--", color="#C8553D", linewidth=1.2)
    ax.set_title("Confidence Interval Trajectories", fontsize=11)
    ax.set_xlabel("total pulls")
    ax.set_ylabel(f"{metric_label} +/- CI")
    ax.grid(alpha=0.2, linestyle=":")
    return _fig_to_base64(fig)


def _build_allocation_plot(pulls_per_arm: list[int]) -> str:
    arr = np.asarray(pulls_per_arm, dtype=np.int_)
    n_arms = int(arr.size)
    idx = np.arange(n_arms)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(idx, arr, color="#1D4E89")
    ax.set_title("Sample Allocation", fontsize=11)
    ax.set_xlabel("arm")
    ax.set_ylabel("pull count")
    ax.set_xticks(idx)
    ax.set_xticklabels([str(i) for i in idx], fontsize=8)
    ax.grid(axis="y", alpha=0.2, linestyle=":")
    return _fig_to_base64(fig)


def _parse_and_validate(form: dict[str, str]) -> tuple[str, str, str, float | None, int, float, str, str, int, int]:
    env_name = form["env"].strip().lower()
    if env_name not in {"bernoulli", "weapon_damage"}:
        raise ValueError("env は bernoulli か weapon_damage を指定してください")

    algo = form["algo"].strip().lower()
    if algo not in {"lucb", "se", "tas"}:
        raise ValueError("algo は lucb / se / tas を指定してください")

    objective = form["objective"].strip().lower()
    if objective not in {"dps", "oneshot"}:
        raise ValueError("objective は dps / oneshot を指定してください")

    threshold: float | None = None
    threshold_raw = form["threshold"].strip()
    if objective == "oneshot":
        if threshold_raw == "":
            raise ValueError("oneshot objective では threshold が必須です")
        try:
            threshold = float(threshold_raw)
        except ValueError as exc:
            raise ValueError("threshold は数値で指定してください") from exc
    elif threshold_raw != "":
        try:
            threshold = float(threshold_raw)
        except ValueError as exc:
            raise ValueError("threshold は数値で指定してください") from exc

    try:
        k = int(form["k"])
    except ValueError as exc:
        raise ValueError("K は整数で指定してください") from exc
    if k < 2 or k > 80:
        raise ValueError("K は 2 以上 80 以下で指定してください")

    try:
        delta = float(form["delta"])
    except ValueError as exc:
        raise ValueError("delta は小数で指定してください") from exc
    if not (0.0 < delta < 1.0):
        raise ValueError("delta は 0 と 1 の間で指定してください")

    means_spec = form["means"].strip()
    if means_spec == "":
        raise ValueError("means regime は空欄にできません")

    weapon_spec = form["pack"].strip()
    if weapon_spec == "":
        raise ValueError("weapon pack regime は空欄にできません")

    try:
        seed = int(form["seed"])
    except ValueError as exc:
        raise ValueError("seed は整数で指定してください") from exc

    try:
        max_pulls = int(form["max_pulls"])
    except ValueError as exc:
        raise ValueError("max_pulls は整数で指定してください") from exc
    if max_pulls < 1000 or max_pulls > 1_000_000:
        raise ValueError("max_pulls は 1000 以上 1000000 以下で指定してください")

    return env_name, algo, objective, threshold, k, delta, means_spec, weapon_spec, seed, max_pulls


def _to_round_preview(history: list[HistoryRecord], take_last: int = 12) -> list[RoundPreview]:
    previews: list[RoundPreview] = []
    for row in history[-take_last:]:
        selected = ",".join(str(arm) for arm in row.selected_arms) if row.selected_arms else "-"
        previews.append(
            RoundPreview(
                round_id=row.round_id,
                total_pulls=row.total_pulls,
                a_hat=row.a_hat,
                selected_arms=selected,
                active_count=len(row.active_arms),
            )
        )
    return previews


def _metric_label(env_name: str, objective: str) -> str:
    if objective == "oneshot":
        return "hit-rate"
    if env_name == "weapon_damage":
        return "expected damage"
    return "mean reward"


def _build_status_table(
    env_name: str,
    objective: str,
    threshold: float | None,
    recommend_arm: int,
    true_values: np.ndarray,
    true_best: int,
    mu_values: np.ndarray | None,
    d0: np.ndarray | None,
    d1: np.ndarray | None,
    p: np.ndarray | None,
    est_counts: np.ndarray,
    est_means: np.ndarray,
    est_lcbs: np.ndarray,
    est_ucbs: np.ndarray,
) -> tuple[list[str], list[list[str]]]:
    order = np.argsort(est_means)[::-1]

    def _marker(arm: int) -> str:
        is_true = arm == true_best
        is_reco = arm == recommend_arm
        if is_true and is_reco:
            return "★◎"
        if is_true:
            return "★"
        if is_reco:
            return "◎"
        return ""

    if env_name == "weapon_damage" and d0 is not None and d1 is not None and p is not None:
        expected = d0 * (1.0 - p) + d1 * p
        oneshot_vals: np.ndarray | None = None
        if threshold is not None:
            hit_d0 = (d0 >= threshold).astype(np.float64)
            hit_d1 = (d1 >= threshold).astype(np.float64)
            oneshot_vals = p * hit_d1 + (1.0 - p) * hit_d0

        if objective == "oneshot":
            headers = ["rank", "mark", "arm", "pulls", "est hit", "CI", "true hit", "d0", "d1", "crit p", "E[dmg]"]
        else:
            headers = ["rank", "mark", "arm", "pulls", "est obj", "CI", "true obj", "d0", "d1", "crit p"]
        rows: list[list[str]] = []
        for rank_idx, arm_idx in enumerate(order, start=1):
            arm = int(arm_idx)
            if objective == "oneshot":
                est_text = f"{100.0 * float(est_means[arm]):.2f}%"
                ci_text = f"[{100.0 * float(est_lcbs[arm]):.2f}%, {100.0 * float(est_ucbs[arm]):.2f}%]"
                true_text = "-" if oneshot_vals is None else f"{100.0 * float(oneshot_vals[arm]):.2f}%"
                rows.append(
                    [
                        str(rank_idx),
                        _marker(arm),
                        str(arm),
                        str(int(est_counts[arm])),
                        est_text,
                        ci_text,
                        true_text,
                        f"{float(d0[arm]):.2f}",
                        f"{float(d1[arm]):.2f}",
                        f"{100.0 * float(p[arm]):.2f}%",
                        f"{float(expected[arm]):.4f}",
                    ]
                )
            else:
                est_text = f"{float(est_means[arm]):.4f}"
                ci_text = f"[{float(est_lcbs[arm]):.4f}, {float(est_ucbs[arm]):.4f}]"
                true_text = f"{float(true_values[arm]):.4f}"
                rows.append(
                    [
                        str(rank_idx),
                        _marker(arm),
                        str(arm),
                        str(int(est_counts[arm])),
                        est_text,
                        ci_text,
                        true_text,
                        f"{float(d0[arm]):.2f}",
                        f"{float(d1[arm]):.2f}",
                        f"{100.0 * float(p[arm]):.2f}%",
                    ]
                )
        return headers, rows

    headers = ["rank", "mark", "arm", "pulls", "est obj", "CI", "true obj", "mu"]
    rows = []
    for rank_idx, arm_idx in enumerate(order, start=1):
        arm = int(arm_idx)
        mu_text = "-" if mu_values is None else f"{float(mu_values[arm]):.4f}"
        rows.append(
            [
                str(rank_idx),
                _marker(arm),
                str(arm),
                str(int(est_counts[arm])),
                f"{float(est_means[arm]):.4f}",
                f"[{float(est_lcbs[arm]):.4f}, {float(est_ucbs[arm]):.4f}]",
                f"{float(true_values[arm]):.4f}",
                mu_text,
            ]
        )
    return headers, rows


def _run_experiment(form: dict[str, str]) -> WebRunResult:
    env_name, algo, objective, threshold, k, delta, means_spec, weapon_spec, seed, max_pulls = _parse_and_validate(form)

    config = RunConfig(
        algo=algo,
        env_name=env_name,
        objective=objective,
        threshold=threshold,
        k=k,
        delta=delta,
        means_spec=means_spec,
        weapon_spec=weapon_spec,
        seed=seed,
        max_pulls=max_pulls,
    )
    problem = build_problem(config, seed=seed)
    algo_impl = build_algo(algo, delta, max_pulls)
    result = algo_impl.run(problem.bandit, track_history=True)
    true_values = problem.true_values
    true_best = problem.true_best

    if not result.history:
        raise RuntimeError("history が空です。実行に失敗しました")

    metric_label = _metric_label(env_name, objective)
    ci_plot = _build_ci_plot(result.history, k, metric_label=metric_label)
    alloc_plot = _build_allocation_plot(result.pulls_per_arm.tolist())
    previews = _to_round_preview(result.history, take_last=12)
    final_state = result.history[-1]
    stop_reason = (
        "max_pulls に到達したため打ち切り停止"
        if result.total_pulls >= max_pulls
        else "信頼区間の停止条件を満たしたため停止"
    )

    top_idx = np.argsort(true_values)[::-1][: min(5, k)]
    top_true_arms = [(int(i), float(true_values[int(i)])) for i in top_idx]
    mu_values: np.ndarray | None = None
    if env_name == "bernoulli":
        base_env = problem.bandit.base_env
        if hasattr(base_env, "means"):
            mu_values = np.asarray(base_env.means, dtype=np.float64)

    status_headers, status_rows = _build_status_table(
        env_name=env_name,
        objective=objective,
        threshold=threshold,
        recommend_arm=result.recommend_arm,
        true_values=true_values,
        true_best=true_best,
        mu_values=mu_values,
        d0=problem.d0,
        d1=problem.d1,
        p=problem.p,
        est_counts=final_state.counts,
        est_means=final_state.means,
        est_lcbs=final_state.lcbs,
        est_ucbs=final_state.ucbs,
    )

    return WebRunResult(
        config=form.copy(),
        recommend_arm=result.recommend_arm,
        true_best=true_best,
        total_pulls=result.total_pulls,
        pulls_per_arm=result.pulls_per_arm.tolist(),
        ci_plot_b64=ci_plot,
        alloc_plot_b64=alloc_plot,
        rounds_preview=previews,
        stop_reason=stop_reason,
        top_true_arms=top_true_arms,
        metric_label=metric_label,
        status_headers=status_headers,
        status_rows=status_rows,
    )


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index() -> Response | str:
        form = dict(DEFAULT_FORM)
        out: WebRunResult | None = None
        error: str | None = None

        if request.method == "POST":
            for key in form:
                if key in request.form:
                    form[key] = str(request.form[key])
            try:
                out = _run_experiment(form)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"

        return render_template_string(HTML_TEMPLATE, form=form, out=out, error=error)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Armory Lab web frontend")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
