from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# ----------------------------
# CP model (2-parameter)
# Work = P*t = W' + CP*t
# Fit linear regression: y = a + b*x  where y=Work(J), x=t(s)
# b=CP (W), a=W' (J)
# ----------------------------

@dataclass
class Interval:
    minutes: int
    seconds: int
    watts: float

    @property
    def t_seconds(self) -> int:
        return int(self.minutes) * 60 + int(self.seconds)


def fit_cp_model(intervals: List[Interval]) -> Tuple[float, float, float]:
    """
    Returns: CP (W), W' (J), R^2
    """
    t = np.array([iv.t_seconds for iv in intervals], dtype=float)
    p = np.array([iv.watts for iv in intervals], dtype=float)

    if np.any(t <= 0):
        raise ValueError("All intervals must have a duration > 0 seconds.")
    if len(intervals) < 2:
        raise ValueError("Enter at least 2 intervals to fit the CP model.")

    work = p * t  # Joules (W*s)

    # Linear regression y = a + b*x
    x = t
    y = work

    x_mean = x.mean()
    y_mean = y.mean()
    sxx = np.sum((x - x_mean) ** 2)
    if sxx == 0:
        raise ValueError("Intervals must have different durations.")

    b = np.sum((x - x_mean) * (y - y_mean)) / sxx  # CP
    a = y_mean - b * x_mean  # W'

    # R^2
    y_hat = a + b * x
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)

    return float(b), float(a), float(r2)


def predict_power(cp_w: float, wprime_j: float, t_seconds: int) -> float:
    if t_seconds <= 0:
        return float("nan")
    return cp_w + (wprime_j / t_seconds)


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Critical Power", layout="wide")

WHITE_THEME_CSS = """
<style>
/* Page background */
.stApp {
  background: #ffffff;
}

/* Make Streamlit blocks breathe a bit */
.block-container {
  padding-top: 1.6rem;
  padding-bottom: 2rem;
}

/* Title row */
.cp-title h1 {
  font-size: 2.2rem;
  margin: 0;
}
.cp-subtitle {
  color: #4b5563;
  margin-top: 0.25rem;
}

/* Card styling */
.cp-card {
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 16px 16px;
  background: #ffffff;
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}
.cp-card h3 {
  margin: 0 0 8px 0;
  font-size: 1.05rem;
}
.cp-muted {
  color: #6b7280;
  font-size: 0.9rem;
}
.cp-metric {
  font-size: 2.2rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin: 4px 0 0 0;
}
.cp-unit {
  font-size: 1rem;
  font-weight: 600;
  color: #6b7280;
  margin-left: 6px;
}
.cp-accent {
  color: #0ea5a8; /* teal-ish accent */
}

/* Interval row */
.interval-row {
  display: grid;
  grid-template-columns: 1fr 1fr 1.3fr 0.4fr;
  gap: 10px;
  align-items: end;
}

/* Small helper text */
.help {
  color: #6b7280;
  font-size: 0.9rem;
}

/* Button tweaks */
div.stButton > button, div.stDownloadButton > button {
  border-radius: 12px;
  padding: 0.6rem 0.9rem;
}
</style>
"""
st.markdown(WHITE_THEME_CSS, unsafe_allow_html=True)

# Initialize intervals
if "intervals" not in st.session_state:
    st.session_state["intervals"] = [
        {"min": 3, "sec": 0, "watts": 417},
        {"min": 12, "sec": 0, "watts": 331},
    ]

# Header (no Export)
st.markdown('<div class="cp-title"><h1>Critical Power</h1></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cp-subtitle">Calculate your physiological threshold (CP) and anaerobic work capacity (W′) using the 2-parameter model.</div>',
    unsafe_allow_html=True,
)

st.write("")
left, right = st.columns([1.05, 1.95], gap="large")

# ----------------------------
# LEFT: Test Intervals
# ----------------------------
with left:
    st.markdown('<div class="cp-card">', unsafe_allow_html=True)
    st.markdown("### Test Intervals")
    st.markdown('<div class="help">Enter your best efforts for different durations.</div>', unsafe_allow_html=True)
    st.write("")

    for idx, row in enumerate(st.session_state["intervals"]):
        cols = st.columns([1, 1, 1.3, 0.6], vertical_alignment="bottom")
        row["min"] = cols[0].number_input("MIN", min_value=0, max_value=240, value=int(row["min"]), key=f"min_{idx}")
        row["sec"] = cols[1].number_input("SEC", min_value=0, max_value=59, value=int(row["sec"]), key=f"sec_{idx}")
        row["watts"] = cols[2].number_input("WATTS", min_value=1, max_value=3000, value=int(row["watts"]), key=f"w_{idx}")
        if cols[3].button("🗑️", key=f"del_{idx}"):
            st.session_state["intervals"].pop(idx)
            st.rerun()
        st.write("")

    add_col, _ = st.columns([1, 1])
    if add_col.button("+  Add Interval"):
        st.session_state["intervals"].append({"min": 5, "sec": 0, "watts": 300})
        st.rerun()

    st.write("")
    st.button(" Calculate Profile", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# RIGHT: Results + Predictor + Plot
# ----------------------------
with right:
    intervals = [
        Interval(int(r["min"]), int(r["sec"]), float(r["watts"]))
        for r in st.session_state["intervals"]
        if (int(r["min"]) * 60 + int(r["sec"])) > 0 and float(r["watts"]) > 0
    ]

    cp_w = None
    wprime_j = None
    r2 = None
    fit_error = None

    if len(intervals) >= 2:
        try:
            cp_w, wprime_j, r2 = fit_cp_model(intervals)
        except Exception as e:
            fit_error = str(e)

    cards = st.columns(3, gap="medium")

    def card(title: str, value: str, subtitle: str):
        st.markdown(
            f"""
            <div class="cp-card">
              <div class="cp-muted">{title}</div>
              <div class="cp-metric">{value}</div>
              <div class="cp-muted">{subtitle}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cards[0]:
        if cp_w is None:
            card("Critical Power (CP)", "—", "Sustainable threshold power")
        else:
            card("Critical Power (CP)", f"{cp_w:.0f}<span class='cp-unit'>W</span>", "Sustainable threshold power")

    with cards[1]:
        if wprime_j is None:
            card("W′ (Anaerobic)", "—", "Energy above threshold")
        else:
            card("W′ (Anaerobic)", f"{(wprime_j/1000):.1f}<span class='cp-unit'>kJ</span>", "Energy above threshold")

    with cards[2]:
        if r2 is None:
            card("Model Fit (R²)", "—", "Statistical accuracy (1.0 is perfect)")
        else:
            card("Model Fit (R²)", f"{r2:.3f}", "Statistical accuracy (1.0 is perfect)")

    if fit_error:
        st.error(f"Model fit issue: {fit_error}")

    st.write("")

    st.markdown('<div class="cp-card">', unsafe_allow_html=True)
    st.markdown("### Performance Predictor")
    st.markdown(
        '<div class="cp-muted">Estimate your max power for any duration based on your CP model.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    pcols = st.columns([1, 1, 0.2, 2], vertical_alignment="bottom")
    pred_min = pcols[0].number_input("MINUTES", min_value=0, max_value=240, value=3)
    pred_sec = pcols[1].number_input("SECONDS", min_value=0, max_value=59, value=0)
    pcols[2].markdown("### ➜")

    pred_t = int(pred_min) * 60 + int(pred_sec)
    if cp_w is not None and wprime_j is not None and pred_t > 0:
        pred_p = predict_power(cp_w, wprime_j, pred_t)
        pcols[3].markdown(
            f"""
            <div class="cp-card" style="border-radius:14px; box-shadow:none;">
              <div class="cp-muted">PREDICTED POWER</div>
              <div class="cp-metric"><span class="cp-accent">{pred_p:.0f}</span><span class="cp-unit">watts</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        pcols[3].markdown(
            """
            <div class="cp-card" style="border-radius:14px; box-shadow:none;">
              <div class="cp-muted">PREDICTED POWER</div>
              <div class="cp-metric">—</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="cp-card">', unsafe_allow_html=True)
    st.markdown("### Power Duration Model")

    fig, ax = plt.subplots()
    if cp_w is not None and wprime_j is not None:
        t_curve = np.linspace(30, 20 * 60, 400)
        p_curve = cp_w + (wprime_j / t_curve)

        ax.plot(t_curve / 60.0, p_curve, linewidth=2)

        ax.axhline(cp_w, linestyle="--", linewidth=1)
        ax.text(20.0, cp_w, f"  CP: {cp_w:.0f}W", va="center")

        t_pts = np.array([iv.t_seconds for iv in intervals]) / 60.0
        p_pts = np.array([iv.watts for iv in intervals])
        ax.scatter(t_pts, p_pts)

        ax.set_xlim(0, 20)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Power (W)")
        ax.grid(True, alpha=0.25)
    else:
        ax.text(0.5, 0.5, "Add at least 2 intervals to see your model curve.", ha="center", va="center")
        ax.set_axis_off()

    st.pyplot(fig, clear_figure=True)
    st.markdown("</div>", unsafe_allow_html=True)
