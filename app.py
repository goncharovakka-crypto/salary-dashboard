"""Дашборд: анализ зарплат в России по видам экономической деятельности (2017–2025)."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Зарплаты в России", page_icon="💰", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
AVG = "В среднем по отраслям"
DEFAULT_SECTORS = ["Информация и связь", "Финансовая и страховая", "Образование",
                   "Сельское и лесное хозяйство", "Здравоохранение и соц. услуги"]
FULL_RANGE = (2017, 2025)


# ---------- данные ----------
@st.cache_data
def load_data():
    wages = pd.read_csv(DATA_DIR / "wages.csv", index_col="Год")
    infl = pd.read_csv(DATA_DIR / "inflation.csv", index_col="Год")["Инфляция, %"]
    return wages, infl


@st.cache_data(ttl=24 * 3600)
def worldbank(indicator: str) -> pd.Series:
    url = (f"https://api.worldbank.org/v2/country/RUS/indicator/{indicator}"
           "?format=json&per_page=200&date=2010:2025")
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    rows = resp.json()[1]
    return pd.Series({int(r["date"]): r["value"] for r in rows if r["value"] is not None}).sort_index()


def compute(wages: pd.DataFrame, infl: pd.Series, y0: int, y1: int):
    """Реальные зарплаты в ценах года y0 и темпы роста к предыдущему году."""
    w = wages.loc[y0:y1]
    inf = infl.loc[y0:y1]
    cum = (1 + inf / 100).cumprod()
    price = cum / cum.loc[y0]                                  # индекс цен, y0 = 1
    real = w.div(price, axis=0)
    nom_g = w.pct_change() * 100
    real_g = ((1 + w.pct_change()).div(1 + inf / 100, axis=0) - 1) * 100
    return w, real, nom_g, real_g, price


wages, infl = load_data()
ALL_SECTORS = [c for c in wages.columns if c != AVG]

# ---------- состояние и боковая панель ----------
if "sel" not in st.session_state:
    st.session_state["sel"] = DEFAULT_SECTORS

def reset_filters():
    st.session_state["sel"] = DEFAULT_SECTORS
    st.session_state["yr"] = FULL_RANGE


st.sidebar.header("⚙️ Настройки")
sectors = st.sidebar.multiselect("Виды деятельности", ALL_SECTORS, key="sel")
y0, y1 = st.sidebar.slider("Период", FULL_RANGE[0], FULL_RANGE[1], value=FULL_RANGE, key="yr")
mode = st.sidebar.radio("Тип зарплаты", ["Номинальная", "Реальная (с учётом инфляции)"])
show_avg = st.sidebar.checkbox("Показать среднее по отраслям", value=True)
log_y = st.sidebar.checkbox("Логарифмическая шкала", value=False)
st.sidebar.button("🔄 Сбросить фильтры", on_click=reset_filters)

# ---------- заголовок ----------
st.title("💰 Зарплаты в России: номинальные и реальные")
st.caption("Данные Росстата по видам экономической деятельности, 2017–2025. "
           "Реальные зарплаты пересчитаны с учётом инфляции.")

if y0 >= y1:
    st.warning("Выберите период минимум из двух лет.")
    st.stop()
if not sectors:
    st.info("Выберите хотя бы один вид деятельности в боковой панели.")
    st.stop()

w, real, nom_g, real_g, price = compute(wages, infl, y0, y1)
is_real = mode.startswith("Реальная")
cols_show = sectors + ([AVG] if show_avg else [])

# ---------- метрики ----------
nom_x = w[AVG].iloc[-1] / w[AVG].iloc[0]
price_x = price.iloc[-1]
real_pct = (real[AVG].iloc[-1] / real[AVG].iloc[0] - 1) * 100
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Средняя зарплата, {y1}", f"{w[AVG].iloc[-1]:,.0f} ₽".replace(",", " "))
c2.metric(f"Номинально, {y0}→{y1}", f"×{nom_x:.2f}")
c3.metric("Рост цен за период", f"×{price_x:.2f}")
c4.metric("Реальный рост", f"{real_pct:+.0f}%")

tab_dyn, tab_growth, tab_rank, tab_extra, tab_data, tab_concl = st.tabs(
    ["📈 Динамика", "📊 Рост к пред. году", "🏆 Рейтинг отраслей",
     "🔍 Доп. исследование", "🗂 Данные", "📝 Выводы"])

# ---------- 1. Динамика (линейный график) ----------
with tab_dyn:
    table = real if is_real else w
    long = table[cols_show].reset_index().melt(id_vars="Год", var_name="Вид деятельности", value_name="руб.")
    title = (f"Реальная зарплата (в ценах {y0} г.), ₽" if is_real else "Номинальная зарплата, ₽")
    fig = px.line(long, x="Год", y="руб.", color="Вид деятельности", markers=True, title=title,
                  log_y=log_y)
    fig.update_layout(legend=dict(orientation="h", y=-0.25), xaxis=dict(dtick=1), height=520)
    st.plotly_chart(fig, use_container_width=True)

    index_view = st.toggle("Показать как индекс (первый год периода = 100)")
    if index_view:
        idx = (table[cols_show] / table[cols_show].iloc[0] * 100).reset_index().melt(
            id_vars="Год", var_name="Вид деятельности", value_name="индекс")
        fig2 = px.line(idx, x="Год", y="индекс", color="Вид деятельности", markers=True,
                       title=f"Индекс ({y0} = 100)")
        fig2.add_hline(y=100, line_dash="dot")
        fig2.update_layout(legend=dict(orientation="h", y=-0.25), xaxis=dict(dtick=1))
        st.plotly_chart(fig2, use_container_width=True)

# ---------- 2. Рост к предыдущему году (столбчатый график) ----------
with tab_growth:
    pick = st.selectbox("Вид деятельности", [AVG] + ALL_SECTORS)
    d = pd.DataFrame({"Номинальный рост, %": nom_g[pick], "Инфляция, %": infl.loc[y0:y1],
                      "Реальный рост, %": real_g[pick]}).dropna()
    fig = go.Figure()
    for col, color in [("Номинальный рост, %", "#4C78A8"), ("Инфляция, %", "#E45756"),
                       ("Реальный рост, %", "#54A24B")]:
        fig.add_bar(x=d.index, y=d[col], name=col, marker_color=color)
    fig.update_layout(barmode="group", title=f"{pick}: номинальный рост, инфляция и реальный рост",
                      xaxis=dict(dtick=1), legend=dict(orientation="h", y=-0.2), height=480)
    fig.add_hline(y=0, line_color="black", line_width=1)
    st.plotly_chart(fig, use_container_width=True)

    neg = d[d["Реальный рост, %"] < 0]
    if neg.empty:
        st.success("За выбранный период реальная зарплата не падала ни в один год.")
    else:
        st.error("Годы падения реальной зарплаты: " +
                 ", ".join(f"{y} ({v:+.1f}%)" for y, v in neg["Реальный рост, %"].items()))
    with st.expander("Таблица значений"):
        st.dataframe(d.round(1), use_container_width=True)

# ---------- 3. Рейтинг отраслей ----------
with tab_rank:
    metric = st.radio("Что сравнивать", ["Реальный рост за период, %", "Номинальный рост за период, %"],
                      horizontal=True)
    base = real if metric.startswith("Реальный") else w
    change = ((base.iloc[-1] / base.iloc[0] - 1) * 100).sort_values()
    avg_change = change[AVG]
    df_r = change.drop(AVG).rename("значение").reset_index().rename(columns={"index": "Вид деятельности"})
    df_r.columns = ["Вид деятельности", "значение"]
    df_r["Выбрано"] = np.where(df_r["Вид деятельности"].isin(sectors), "Выбрано", "Остальные")
    fig = px.bar(df_r, x="значение", y="Вид деятельности", orientation="h", color="Выбрано",
                 color_discrete_map={"Выбрано": "#F58518", "Остальные": "#9ecae9"},
                 title=f"{metric} ({y0}→{y1})", height=640)
    fig.add_vline(x=avg_change, line_dash="dash", annotation_text="среднее по отраслям")
    st.plotly_chart(fig, use_container_width=True)
    best, worst = df_r.iloc[-1], df_r.iloc[0]
    st.write(f"**Лидер:** {best['Вид деятельности']} ({best['значение']:+.0f}%). "
             f"**Аутсайдер:** {worst['Вид деятельности']} ({worst['значение']:+.0f}%).")

# ---------- 4. Дополнительное исследование ----------
with tab_extra:
    st.write("Связь реальной зарплаты (в среднем по отраслям) с макропоказателями. "
             "Данные Всемирного банка загружаются по кнопке.")
    indicators = {"Безработица, %": "SL.UEM.TOTL.ZS", "Рост ВВП, %": "NY.GDP.MKTP.KD.ZG",
                  "ВВП на душу населения, $": "NY.GDP.PCAP.CD"}
    if st.button("⬇️ Загрузить данные Всемирного банка"):
        try:
            with st.spinner("Загружаю..."):
                st.session_state["macro"] = pd.DataFrame({k: worldbank(v) for k, v in indicators.items()})
            st.success("Данные загружены.")
        except Exception as exc:  # сеть недоступна и т.п.
            st.error(f"Не удалось загрузить данные: {exc}")

    if "macro" in st.session_state:
        macro = st.session_state["macro"]
        _, real_full, _, real_g_full, _ = compute(wages, infl, *FULL_RANGE)
        ind = st.selectbox("Показатель", list(indicators))
        target = st.radio("Что сравнивать с показателем",
                          ["Индекс реальной зарплаты (2017 = 100)", "Реальный рост зарплаты, % к пред. году"],
                          horizontal=True)
        y_series = (real_full[AVG] / real_full[AVG].iloc[0] * 100 if target.startswith("Индекс")
                    else real_g_full[AVG])
        d = pd.concat([macro[ind], y_series.rename("y")], axis=1, join="inner").dropna()
        if len(d) < 4:
            st.warning("Слишком мало общих наблюдений.")
        else:
            r = d[ind].corr(d["y"])
            k, b = np.polyfit(d[ind], d["y"], 1)
            fig = px.scatter(d.reset_index().rename(columns={"index": "Год"}), x=ind, y="y",
                             text="Год", labels={"y": target}, title=f"{ind} и реальная зарплата (r = {r:.2f})")
            xs = np.linspace(d[ind].min(), d[ind].max(), 20)
            fig.add_trace(go.Scatter(x=xs, y=k * xs + b, mode="lines", name="линейный тренд",
                                     line=dict(dash="dash")))
            fig.update_traces(textposition="top center", selector=dict(mode="markers+text"))
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Коэффициент корреляции Пирсона", f"{r:.2f}", help=f"Наблюдений: {len(d)}")
            st.caption("⚠️ Корреляция не означает причинность; выборка небольшая (годовые данные).")

# ---------- 5. Данные ----------
with tab_data:
    st.subheader("Зарплаты, руб.")
    st.dataframe(wages.T, use_container_width=True)
    st.subheader("Инфляция, % (декабрь к декабрю)")
    st.dataframe(infl.to_frame().T, use_container_width=True)
    st.download_button("💾 Скачать зарплаты (CSV)", wages.to_csv().encode("utf-8-sig"),
                       file_name="wages_2017_2025.csv", mime="text/csv")

# ---------- 6. Выводы ----------
with tab_concl:
    st.markdown("""
**Основные выводы (2017–2025):**

- Цены выросли в 1,7 раза, а номинальная зарплата в среднем по отраслям — в 2,56 раза.
  Реальные зарплаты выросли примерно на 51%.
- В среднем по экономике реальные зарплаты росли каждый год; слабее всего — в 2020–2022 годах (+2–3%).
- В отдельных отраслях бывало падение: в 2020 году (например, гостиницы и общепит, −7,3%) и в 2021–2022 годах
  в бюджетных сферах (госуправление, здравоохранение, образование).
- Лидеры реального роста: информация и связь (+83%), сельское хозяйство (+73%), строительство (+72%);
  аутсайдеры: госуправление (+29%), энергетика (+31%), добыча полезных ископаемых (+32%).
- Разрыв между самой высокооплачиваемой и самой низкооплачиваемой отраслью — около 3,5 раза.

*Подробности смотрите на вкладках выше — выводы можно проверить на графиках при любых настройках фильтров.*
""")

st.sidebar.markdown("---")
st.sidebar.caption("Источники: Росстат; уровень-инфляции.рф; Всемирный банк.")
