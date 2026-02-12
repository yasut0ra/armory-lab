from __future__ import annotations

import argparse
import base64
import io
from dataclasses import dataclass

import matplotlib
import numpy as np
from flask import Flask, Response, render_template_string, request
from numpy.typing import NDArray

from armory_lab.algos.base import BAIResult, HistoryRecord
from armory_lab.envs.bernoulli import BernoulliBandit
from armory_lab.run import build_algo, generate_means

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(slots=True)
class WebRunResult:
    config: dict[str, str]
    result: BAIResult
    means: NDArray[np.float64]
    true_best: int
    ci_plot_b64: str
    alloc_plot_b64: str
    rounds_preview: list[HistoryRecord]


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
      --ink: #121212;
      --card: #fffdf9;
      --muted: #57534e;
      --line: #d6d3d1;
      --brand-a: #c8553d;
      --brand-b: #1d4e89;
      --brand-c: #f2b705;
      --ok: #1a7f37;
      --bad: #ab1f1f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Space Grotesk", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 8% 12%, #f6cf9f66 0, transparent 34%),
        radial-gradient(circle at 84% 88%, #a8c9ee70 0, transparent 30%),
        linear-gradient(155deg, #f7f4ed 0%, #efe9dd 40%, #e9f0f8 100%);
      min-height: 100vh;
    }
    .wrap {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px 20px 48px;
      animation: rise .45s ease-out;
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }
    h1 {
      margin: 0 0 8px;
      font-size: clamp(1.7rem, 3vw, 2.3rem);
      letter-spacing: .01em;
    }
    .lead {
      color: var(--muted);
      margin-bottom: 22px;
    }
    .grid {
      display: grid;
      grid-template-columns: 350px 1fr;
      gap: 18px;
      align-items: start;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 10px 24px #0000000e;
    }
    .stack {
      display: grid;
      gap: 12px;
    }
    label {
      display: block;
      font-size: .9rem;
      color: var(--muted);
      margin-bottom: 4px;
    }
    input, select {
      width: 100%;
      border: 1px solid #cfc8be;
      border-radius: 10px;
      background: #fff;
      color: #1f1f1f;
      font-size: 0.95rem;
      padding: 9px 10px;
      outline: none;
    }
    input:focus, select:focus {
      border-color: var(--brand-b);
      box-shadow: 0 0 0 3px #1d4e8920;
    }
    button {
      width: 100%;
      border: 0;
      border-radius: 12px;
      padding: 11px 12px;
      background: linear-gradient(105deg, var(--brand-a), #df6f53);
      color: #fff;
      font-size: 0.98rem;
      font-weight: 700;
      cursor: pointer;
      transition: transform .12s ease, filter .12s ease;
    }
    button:hover {
      transform: translateY(-1px);
      filter: saturate(1.08);
    }
    .mono { font-family: "JetBrains Mono", monospace; }
    .row {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric {
      border-left: 4px solid var(--brand-b);
      padding: 10px 10px 10px 12px;
      background: #fbfaf6;
      border-radius: 10px;
    }
    .metric .k {
      font-size: .78rem;
      color: #67615a;
      margin-bottom: 4px;
    }
    .metric .v {
      font-size: 1.1rem;
      font-weight: 700;
    }
    .ok { color: var(--ok); }
    .bad { color: var(--bad); }
    .charts {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }
    .charts img {
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #fff;
    }
    .table-wrap {
      margin-top: 12px;
      max-height: 300px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: .86rem;
    }
    thead th {
      text-align: left;
      position: sticky;
      top: 0;
      background: #faf7f1;
      border-bottom: 1px solid var(--line);
      padding: 8px;
    }
    tbody td {
      border-bottom: 1px solid #f0ece4;
      padding: 8px;
      vertical-align: top;
    }
    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
      .row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <main class="wrap">
    <h1>Armory Lab BAI Console</h1>
    <p class="lead">fixed-confidence BAIで「最強武器」を鑑定するブラウザUI。LUCB/Successive Eliminationを即比較できます。</p>

    <section class="grid">
      <form class="card stack" method="post">
        <div>
          <label for="algo">Algorithm</label>
          <select id="algo" name="algo">
            <option value="lucb" {% if form.algo == "lucb" %}selected{% endif %}>LUCB</option>
            <option value="se" {% if form.algo == "se" %}selected{% endif %}>Successive Elimination</option>
          </select>
        </div>
        <div>
          <label for="k">K (arms)</label>
          <input id="k" name="k" type="number" min="2" max="80" value="{{ form.k }}" />
        </div>
        <div>
          <label for="delta">delta</label>
          <input id="delta" name="delta" type="number" step="0.001" min="0.001" max="0.999" value="{{ form.delta }}" />
        </div>
        <div>
          <label for="means">means regime</label>
          <input id="means" name="means" type="text" value="{{ form.means }}" />
        </div>
        <div>
          <label for="seed">seed</label>
          <input id="seed" name="seed" type="number" value="{{ form.seed }}" />
        </div>
        <div>
          <label for="max_pulls">max pulls</label>
          <input id="max_pulls" name="max_pulls" type="number" min="1000" max="1000000" value="{{ form.max_pulls }}" />
        </div>
        <button type="submit">Run Appraisal</button>
        <div class="mono" style="font-size:.8rem;color:#6b6460">means例: <code>random</code>, <code>two-groups</code>, <code>topgap:0.07</code></div>
      </form>

      <section class="card">
        {% if error %}
          <div class="bad mono">{{ error }}</div>
        {% elif out %}
          <div class="row">
            <article class="metric">
              <div class="k">recommended arm</div>
              <div class="v mono">{{ out.result.recommend_arm }}</div>
            </article>
            <article class="metric">
              <div class="k">true best arm</div>
              <div class="v mono">{{ out.true_best }}</div>
            </article>
            <article class="metric">
              <div class="k">correct</div>
              <div class="v {% if out.result.recommend_arm == out.true_best %}ok{% else %}bad{% endif %}">
                {% if out.result.recommend_arm == out.true_best %}YES{% else %}NO{% endif %}
              </div>
            </article>
            <article class="metric">
              <div class="k">total pulls</div>
              <div class="v mono">{{ out.result.total_pulls }}</div>
            </article>
          </div>

          <div class="mono" style="font-size:.84rem;margin-bottom:10px;color:#5e5852">
            config: algo={{ out.config["algo"] }} / K={{ out.config["k"] }} / delta={{ out.config["delta"] }} / means={{ out.config["means"] }} / seed={{ out.config["seed"] }}
          </div>

          <div class="charts">
            <img src="data:image/png;base64,{{ out.ci_plot_b64 }}" alt="CI trajectory">
            <img src="data:image/png;base64,{{ out.alloc_plot_b64 }}" alt="Allocation">
          </div>

          <h3 style="margin:14px 0 8px;font-size:1rem">Recent rounds</h3>
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
                  <td>{{ row.active_arms|length }}</td>
                </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        {% else %}
          <div style="color:#645f57">左の設定を調整して <span class="mono">Run Appraisal</span> を押すと、結果と可視化を表示します。</div>
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


def _build_ci_plot(history: list[HistoryRecord], n_arms: int) -> str:
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
    ax.set_ylabel("mean +/- CI")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.2, linestyle=":")
    return _fig_to_base64(fig)


def _build_allocation_plot(result: BAIResult) -> str:
    n_arms = int(result.pulls_per_arm.size)
    idx = np.arange(n_arms)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(idx, result.pulls_per_arm, color="#1D4E89")
    ax.set_title("Sample Allocation", fontsize=11)
    ax.set_xlabel("arm")
    ax.set_ylabel("pull count")
    ax.set_xticks(idx)
    ax.set_xticklabels([str(i) for i in idx], fontsize=8)
    ax.grid(axis="y", alpha=0.2, linestyle=":")
    return _fig_to_base64(fig)


def _run_experiment(form: dict[str, str]) -> WebRunResult:
    algo = form["algo"]
    k = int(form["k"])
    delta = float(form["delta"])
    means_spec = form["means"]
    seed = int(form["seed"])
    max_pulls = int(form["max_pulls"])

    rng = np.random.default_rng(seed)
    means = generate_means(means_spec, k, rng)
    env = BernoulliBandit.from_means(means, rng=rng)
    algo_impl = build_algo(algo, delta, max_pulls)
    result = algo_impl.run(env, track_history=True)
    true_best = int(np.argmax(means))

    if not result.history:
        raise RuntimeError("history is empty; cannot render plots")

    ci_plot = _build_ci_plot(result.history, k)
    alloc_plot = _build_allocation_plot(result)
    preview = result.history[-12:]
    return WebRunResult(
        config=form,
        result=result,
        means=means,
        true_best=true_best,
        ci_plot_b64=ci_plot,
        alloc_plot_b64=alloc_plot,
        rounds_preview=preview,
    )


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index() -> Response | str:
        form = {
            "algo": "lucb",
            "k": "20",
            "delta": "0.05",
            "means": "topgap:0.05",
            "seed": "0",
            "max_pulls": "200000",
        }
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
