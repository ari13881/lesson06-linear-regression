import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업의 기준 기간 끝
MIN_DAYS = 300        # 관측일이 이보다 적은 해는 제외

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="centered")


@st.cache_data(show_spinner="서울 기온 데이터를 불러오는 중...")
def load_yearly_mean() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]

    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"]).drop_duplicates(subset="날짜")

    df["연도"] = df["날짜"].dt.year
    df = df[df["연도"] <= LAST_YEAR]  # 2025년 이후 데이터 제외

    yearly = (
        df.groupby("연도")["평균기온"]
        .agg(연평균기온="mean", 관측일수="count")
        .reset_index()
    )
    yearly = yearly[yearly["관측일수"] >= MIN_DAYS]  # 관측일 300일 미만인 해 제외
    return yearly.reset_index(drop=True)


st.title("🌡️ 기온 예측기")
st.caption("서울의 연평균기온 추세를 직선으로 맞춰 원하는 해의 기온을 예상해 봅니다.")

try:
    yearly = load_yearly_mean()
except Exception as e:  # 네트워크/형식 오류 안내
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

if len(yearly) < 2:
    st.warning("회귀 직선을 만들 데이터가 부족합니다.")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# 최근 20년(남은 데이터의 마지막 20년 구간)만으로 따로 구한 직선
RECENT_SPAN = 20
recent = yearly[yearly["연도"] > end_year - RECENT_SPAN]
recent_n = len(recent)
recent_start = int(recent["연도"].min())
if recent_n >= 2:
    recent_slope, recent_intercept = np.polyfit(
        recent["연도"].to_numpy(dtype=float),
        recent["연평균기온"].to_numpy(dtype=float),
        1,
    )
else:
    recent_slope, recent_intercept = None, None

full_per_100 = slope * 100


def big_card(title: str, value: str, sub: str) -> str:
    return f"""
    <div style="text-align:center; padding:0.8rem 0.4rem; border:1px solid rgba(128,128,128,0.35);
                border-radius:12px;">
        <div style="font-size:0.95rem; opacity:0.75;">{title}</div>
        <div style="font-size:2.6rem; font-weight:800; line-height:1.25;">{value}</div>
        <div style="font-size:0.85rem; opacity:0.65;">{sub}</div>
    </div>
    """


# ── 사용한 데이터 안내 ──────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("직선을 만든 해의 개수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")

# ── 연도 슬라이더와 예상 기온 ───────────────────────
year = st.slider("예상할 연도를 고르세요", min_value=1900, max_value=2100, value=2050, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem 0;">
        <div style="font-size:1.1rem; opacity:0.75;">{year}년의 예상 평균기온</div>
        <div style="font-size:4.5rem; font-weight:800; line-height:1.2;">{pred:.1f} °C</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 산점도 + 회귀 직선 ──────────────────────────────
line_x = np.array([min(1900, start_year), 2100], dtype=float)
line_y = slope * line_x + intercept

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=7),
        hovertemplate="%{x}년<br>%{y:.2f} °C<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3),
        hovertemplate="%{x:.0f}년<br>%{y:.2f} °C<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예상",
        marker=dict(size=16, symbol="star", line=dict(width=1, color="black")),
        hovertemplate="%{x}년 예상<br>%{y:.2f} °C<extra></extra>",
    )
)
if recent_slope is not None:
    rx = np.array([recent_start, end_year], dtype=float)
    fig.add_trace(
        go.Scatter(
            x=rx,
            y=recent_slope * rx + recent_intercept,
            mode="lines",
            name=f"최근 {RECENT_SPAN}년 직선",
            line=dict(width=3, dash="dash"),
            hovertemplate="%{x:.0f}년<br>%{y:.2f} °C<extra></extra>",
        )
    )
fig.update_layout(
    title="서울 연평균기온과 회귀 직선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    xaxis=dict(range=[1895, 2105]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=80, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# ── 상관계수와 직선 식 ──────────────────────────────
st.metric("상관계수 (연도와 연평균기온)", f"{r:.3f}")
st.caption(f"회귀 직선(전체 기간): 기온 = {slope:.4f} × 연도 + ({intercept:.2f})")

# ── 기울기: 100년에 몇 도 오르는가 ───────────────────
st.subheader("100년에 몇 도 오를까?")
st.markdown(
    f"""
    <div style="text-align:center; padding:0.8rem 0;">
        <div style="font-size:1.05rem; opacity:0.75;">전체 기간({start_year}~{end_year}년) 직선 기준</div>
        <div style="font-size:4rem; font-weight:800; line-height:1.2;">{full_per_100:+.1f} °C / 100년</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("##### 전체 기간 vs 최근 20년")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(
        big_card(
            "전체 기간",
            f"{full_per_100:+.1f} °C",
            f"100년당 · {start_year}~{end_year}년 · {n_years}개 해",
        ),
        unsafe_allow_html=True,
    )
with col_b:
    if recent_slope is not None:
        st.markdown(
            big_card(
                f"최근 {RECENT_SPAN}년",
                f"{recent_slope * 100:+.1f} °C",
                f"100년당 · {recent_start}~{end_year}년 · {recent_n}개 해",
            ),
            unsafe_allow_html=True,
        )
    else:
        st.warning("최근 20년 구간에 쓸 수 있는 해가 부족합니다.")

if recent_slope is not None and full_per_100 != 0:
    diff = recent_slope * 100 - full_per_100
    ratio = recent_slope / slope
    st.write(
        f"최근 {RECENT_SPAN}년의 상승 속도는 전체 기간보다 100년당 **{diff:+.1f} °C** 차이가 나며, "
        f"전체 기간의 약 **{ratio:.1f}배**입니다."
    )
st.caption(
    f"'100년당' 값은 연간 기울기에 100을 곱한 것입니다. 최근 {RECENT_SPAN}년은 해가 {recent_n}개뿐이라 "
    "기울기가 해마다의 들쭉날쭉함에 크게 흔들릴 수 있습니다."
)

st.info(
    f"기준 기간은 {LAST_YEAR}년까지이며, 관측일이 {MIN_DAYS}일 미만인 해는 제외했습니다. "
    "직선을 만든 기간 밖의 연도(특히 먼 미래)는 추세를 그대로 늘린 값이라 참고용으로만 보세요."
)
