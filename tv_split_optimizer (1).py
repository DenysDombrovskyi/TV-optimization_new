import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import plotly.express as px

# -----------------------------
#            Utils
# -----------------------------

TOP_CHANNEL_GROUPS = {
    "Оушен": ["СТБ", "Новий канал", "ICTV2"],
    "Sirius": ["1+1 Україна", "ТЕТ", "2+2"],  # ✅ ТЕТ у Sirius
    "Space": ["НТН"],
}
ALL_TOP_CHANNELS = [ch for lst in TOP_CHANNEL_GROUPS.values() for ch in lst]

st.set_page_config(page_title="Оптимізація ТВ спліта", layout="wide")
st.title("📺 Евристична оптимізація ТВ спліта | Dentsu X")

# -----------------------------
#       Validation helpers
# -----------------------------

def validate_structure(df: pd.DataFrame) -> bool:
    """Базова перевірка колонок."""
    base_required = ["Канал", "СХ"]
    for col in base_required:
        if col not in df.columns:
            st.error(f"❌ В аркуші 'Сп-во' відсутній обов'язковий стовпчик '{col}'.")
            return False
    return True

def get_available_ba(df: pd.DataFrame) -> list:
    """Знаходимо всі доступні BA за префіксами колонок."""
    price_bas = [c.replace("Ціна_", "") for c in df.columns if c.startswith("Ціна_")]
    trp_bas   = [c.replace("TRP_", "") for c in df.columns if c.startswith("TRP_")]
    aff_bas   = [c.replace("Affinity_", "") for c in df.columns if c.startswith("Affinity_")]
    # лишаємо лише ті BA, які мають весь набір колонок
    all_ba = sorted(set(price_bas) & set(trp_bas) & set(aff_bas))
    return all_ba

def validate_ba_columns_exist(df: pd.DataFrame, buying_audiences: dict) -> bool:
    """Переконуємось, що для всіх вибраних СХ існують колонки Ціна/TRP/Affinity."""
    ok = True
    for sh, ba in buying_audiences.items():
        missing = [col for col in (f"Ціна_{ba}", f"TRP_{ba}", f"Affinity_{ba}") if col not in df.columns]
        if missing:
            st.error(f"❌ Для СХ '{sh}' відсутні колонки для БА '{ba}': {', '.join(missing)}")
            ok = False
    return ok

def coerce_numeric_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Перетворюємо в числові з NaN -> 0 (для стабільності)."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# -----------------------------
#        Core logic
# -----------------------------

def heuristic_split_within_group(group_df: pd.DataFrame, total_group_budget: float) -> pd.DataFrame:
    """
    Евристичний розподіл бюджету всередині підгрупи каналів:
    пріоритет — мінімальна 'цільова вартість TRP' (Ціна/TRP * Affinity).
    Бюджет на канал обмежено його 'потенціалом' = Ціна * TRP.
    """
    group_df = group_df.copy()

    if group_df.empty or total_group_budget <= 0:
        group_df["Оптимальний бюджет"] = 0.0
        group_df["Оптимальна частка (%)"] = 0.0
        return group_df

    # Вартість покупного TRP з обробкою ділення на нуль
    cost_per_buying_trp = np.divide(
        group_df["Ціна"], group_df["TRP"],
        out=np.full_like(group_df["TRP"].to_numpy(dtype=float), np.inf, dtype=float),
        where=group_df["TRP"].to_numpy(dtype=float) != 0
    )
    # Вартість цільового TRP
    cost_per_targeted_trp = cost_per_buying_trp * group_df["Affinity"].fillna(1.0)

    # Розподіляємо бюджет у порядку зростання cost_per_targeted_trp
    sorted_idx = cost_per_targeted_trp.sort_values(ascending=True).index
    shares = pd.Series(0.0, index=group_df.index)
    remaining = float(total_group_budget)

    for idx in sorted_idx:
        row = group_df.loc[idx]
        cost = cost_per_targeted_trp.loc[idx]
        if pd.notna(cost) and np.isfinite(cost):
            capacity = float(row["Ціна"]) * float(row["TRP"])  # максимум, який "містить" канал
            add = min(remaining, capacity)
            shares.loc[idx] = add
            remaining -= add
            if remaining <= 1e-9:
                break

    group_df["Оптимальний бюджет"] = shares
    # Частки всередині СХ рахуватимемо пізніше, коли знатимемо total_sx_budget
    return group_df

def run_multi_group_optimization(df: pd.DataFrame, buying_audiences: dict) -> pd.DataFrame:
    """
    Головна оптимізація:
    1) Для кожного СХ фіксуємо бюджети для кожної топ-групи (= поточна сума Ціна*TRP у цій підмножині)
    2) Розподіляємо їх усередині груп за евристикою
    3) Решту каналів оптимізуємо окремо на їхній сумі
    4) Обчислюємо частки від total_sx_budget і нормалізуємо до 100% (зберігаючи пропорції)
    """
    df = df.copy()

    # Підтягнути вибрану БА для кожного рядка
    df["Ціна"] = df.apply(lambda r: r.get(f"Ціна_{buying_audiences.get(r['СХ'], '')}", 0), axis=1)
    df["TRP"] = df.apply(lambda r: r.get(f"TRP_{buying_audiences.get(r['СХ'], '')}", 0), axis=1)
    df["Affinity"] = df.apply(lambda r: r.get(f"Affinity_{buying_audiences.get(r['СХ'], '')}", 1.0), axis=1).fillna(1.0)

    coerce_numeric_cols(df, ["Ціна", "TRP", "Affinity"])

    all_results = []

    for sh, sh_df in df.groupby("СХ", sort=False):
        sh_df = sh_df.copy()
        total_sx_budget = float((sh_df["Ціна"] * sh_df["TRP"]).sum())

        # Розбиваємо на топ-підмножини та "решту"
        remaining_df = sh_df.copy()
        optimized_parts = []

        for group_name, channels in TOP_CHANNEL_GROUPS.items():
            part = remaining_df[remaining_df["Канал"].isin(channels)].copy()
            if not part.empty:
                part_budget = float((part["Ціна"] * part["TRP"]).sum())  # фіксований бюджет групи
                part_opt = heuristic_split_within_group(part, part_budget)
                optimized_parts.append(part_opt)
                # Викидаємо оптимізовані підмножини з “решти”
                remaining_df = remaining_df[~remaining_df["Канал"].isin(channels)]

        if not remaining_df.empty:
            rest_budget = float((remaining_df["Ціна"] * remaining_df["TRP"]).sum())
            rest_opt = heuristic_split_within_group(remaining_df, rest_budget)
            optimized_parts.append(rest_opt)

        sh_result = pd.concat(optimized_parts, axis=0) if optimized_parts else sh_df.copy()
        # частки від total_sx_budget
        if total_sx_budget > 0:
            sh_result["Оптимальна частка (%)"] = sh_result["Оптимальний бюджет"] / total_sx_budget * 100.0
        else:
            sh_result["Оптимальна частка (%)"] = 0.0

        all_results.append(sh_result)

    all_results = pd.concat(all_results, axis=0)

    # Нормалізуємо по кожному СХ до 100% (зберігаючи пропорції — отже, сумарні частки топів не зміняться)
    def _norm(x: pd.Series) -> pd.Series:
        s = x.sum()
        return x if s == 0 else x / s * 100.0

    all_results["Оптимальна частка (%)"] = all_results.groupby("СХ")["Оптимальна частка (%)"].transform(_norm)

    # Обчислюємо службову метрику: вартість цільового TRP (для візуалізацій/виділень)
    with np.errstate(divide="ignore", invalid="ignore"):
        all_results["Ціна_ТРП_цільовий"] = (
            np.divide(
                all_results["Ціна"], all_results["TRP"],
                out=np.full_like(all_results["TRP"].to_numpy(dtype=float), np.inf, dtype=float),
                where=all_results["TRP"].to_numpy(dtype=float) != 0
            ) * all_results["Affinity"].fillna(1.0)
        )

    return all_results

# -----------------------------
#         Demo data
# -----------------------------

def make_demo_df() -> pd.DataFrame:
    """Створює тестовий датафрейм, що імітує аркуш 'Сп-во'."""
    data = {
        "Канал": [
            "СТБ", "Новий канал", "ICTV2", "1+1 Україна", "ТЕТ", "2+2", "НТН",
            "ICTV", "Україна 24", "ПлюсПлюс"
        ],
        "СХ": [
            "FMCG", "FMCG", "FMCG", "FMCG", "FMCG", "FMCG", "FMCG",
            "Finance", "Finance", "Finance"
        ],
        # BA = A18-54
        "Ціна_A18-54": [100, 110, 95, 130, 90, 85, 70, 120, 140, 60],
        "TRP_A18-54":  [15,  12,  11,  14, 13, 9,   8,  10,  8,   7 ],
        "Affinity_A18-54": [1.0, 1.05, 0.95, 1.1, 0.9, 1.0, 0.85, 1.0, 1.1, 0.9],
        # BA = W25-45
        "Ціна_W25-45": [90, 100, 92, 120, 88, 80, 75, 115, 135, 65],
        "TRP_W25-45":  [14, 11,  12, 13,  12, 8,  9,  9,   8,   7],
        "Affinity_W25-45": [1.1, 1.0, 1.0, 1.05, 0.95, 0.9, 0.9, 1.0, 1.1, 0.95],
    }
    df = pd.DataFrame(data)
    return df

# -----------------------------
#         UI: Input
# -----------------------------

st.sidebar.header("Дані")
use_demo = st.sidebar.toggle("🧪 Використати демо-дані (без Excel)", value=False)

df_input = None
if use_demo:
    df_input = make_demo_df()
    st.success("✅ Використовуються демо-дані.")
else:
    uploaded = st.file_uploader("Завантажте Excel-файл з аркушем 'Сп-во'", type=["xlsx"])
    if uploaded is not None:
        try:
            # Зберігаємо вихідний порядок рядків
            df_input = pd.read_excel(uploaded, sheet_name="Сп-во", skiprows=2, engine="openpyxl")
            st.success("✅ Дані успішно завантажено!")
        except Exception as e:
            st.error(f"❌ Помилка при завантаженні файлу: {e}")

if df_input is None:
    st.info("⬆️ Завантажте файл або увімкніть демо-дані в лівій панелі.")
    st.stop()

# Зберігаємо оригінальний порядок каналів
df_input = df_input.copy()
df_input["Порядок"] = np.arange(len(df_input))

# Перевірки
if not validate_structure(df_input):
    st.stop()

available_ba = get_available_ba(df_input)
if not available_ba:
    st.error("❌ У файлі немає повного набору колонок для жодної БА (потрібні Ціна_*, TRP_*, Affinity_*).")
    st.stop()

st.header("🔧 Налаштування оптимізації")
st.subheader("🎯 Вибір Buying Audience (БА) для кожного СХ")

buying_audiences = {}
col_left, col_right = st.columns([1, 1])
with col_left:
    all_sh = list(df_input["СХ"].unique())
    for sh in all_sh:
        ba = st.selectbox(f"СХ: {sh}", available_ba, key=f"ba_{sh}")
        buying_audiences[sh] = ba

if not validate_ba_columns_exist(df_input, buying_audiences):
    st.stop()

st.subheader("📊 Топ-групи каналів (застосовуються до кожного СХ)")
st.caption("Бюджет кожної топ-групи фіксується як поточна сума Ціна*TRP каналів цієї групи в межах СХ і розподіляється всередині групи.")
st.write(
    pd.DataFrame(
        [(g, ", ".join(chs)) for g, chs in TOP_CHANNEL_GROUPS.items()],
        columns=["Група", "Канали"]
    )
)

# -----------------------------
#         RUN OPTIMIZATION
# -----------------------------

run = st.button("🚀 Запустити оптимізацію")

if run:
    # Розрахунок
    df_work = df_input.copy()

    # Впорядкуємо групування за вихідним порядком
    # (groupby(sort=False) + збереження "Порядок" дозволяє виводити як в Excel)
    results = run_multi_group_optimization(df_work, buying_audiences)

    # Повертаємо вихідний порядок відображення
    results = results.merge(df_work[["Канал", "СХ", "Порядок"]], on=["Канал", "СХ"], how="left")
    results = results.sort_values(["СХ", "Порядок"], kind="stable").reset_index(drop=True)

    # Додаткові службові колонки
    results["Початковий бюджет"] = results["Ціна"] * results["TRP"]
    results["is_top"] = results["Канал"].isin(ALL_TOP_CHANNELS)

    # -----------------------------
    #         TABLES OUTPUT
    # -----------------------------

    st.subheader("📊 Результати оптимізації по СХ (у вихідному порядку)")
    for sh in results["СХ"].unique():
        st.markdown(f"#### СХ: {sh}")
        sh_df = results[results["СХ"] == sh].copy()

        # Відображення детальної таблиці
        cols_show = [
            "Канал", "Початковий бюджет", "Оптимальний бюджет",
            "Ціна_ТРП_цільовий", "Оптимальна частка (%)"
        ]
        st.dataframe(
            sh_df[cols_show],
            use_container_width=True
        )

    # -----------------------------
    #      TOP SUMMARY (per СХ)
    # -----------------------------

    st.subheader("📌 Сумарні частки Топ-каналів по СХ (після нормалізації)")
    top_share_summary = (
        results[results["is_top"]]
        .groupby("СХ", sort=False)["Оптимальна частка (%)"]
        .sum()
        .reset_index()
        .rename(columns={"Оптимальна частка (%)": "Сумарна частка Топ-каналів (%)"})
    )

    # Список топ-каналів у порядку появи в кожному СХ
    top_channels_per_sh = (
        results[results["is_top"]]
        .sort_values(["СХ", "Порядок"], kind="stable")
        .groupby("СХ")["Канал"]
        .apply(lambda s: ", ".join(s.drop_duplicates()))
        .reset_index()
        .rename(columns={"Канал": "Топ-канали"})
    )

    top_summary = pd.merge(top_share_summary, top_channels_per_sh, on="СХ", how="outer")
    st.dataframe(top_summary, use_container_width=True)

    # -----------------------------
    #           CHARTS
    # -----------------------------

    st.subheader("📈 Графіки сплітів по кожному СХ")

    for sh in results["СХ"].unique():
        sh_df = results[results["СХ"] == sh].copy()
        # для наочності — категоріальний порядок як у файлі
        sh_df["Канал"] = pd.Categorical(sh_df["Канал"], categories=list(sh_df["Канал"]), ordered=True)

        fig = px.bar(
            sh_df,
            x="Канал",
            y="Оптимальна частка (%)",
            color=sh_df["is_top"].map({True: "Топ-канал", False: "Інший"}),
            hover_data={
                "Оптимальна частка (%)": ":.2f",
                "Ціна_ТРП_цільовий": ":.2f",
                "Початковий бюджет": ":.2f",
                "Оптимальний бюджет": ":.2f",
                "Канал": True,
                "is_top": False
            },
            barmode="group",
            title=f"СХ: {sh} — оптимальна частка по каналах"
        )
        fig.update_layout(xaxis_title="", yaxis_title="Частка (%)")
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    #           EXPORT
    # -----------------------------

    st.subheader("📥 Експорт у Excel")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Детальні результати
        cols_order = [
            "СХ", "Порядок", "Канал", "Ціна", "TRP", "Affinity",
            "Початковий бюджет", "Оптимальний бюджет", "Оптимальна частка (%)", "Ціна_ТРП_цільовий", "is_top"
        ]
        results[cols_order].to_excel(writer, sheet_name="Оптимальний спліт", index=False)
        top_summary.to_excel(writer, sheet_name="Сумарні Топ-частки", index=False)

    st.download_button(
        "⬇️ Завантажити результати (Excel)",
        data=output.getvalue(),
        file_name="результати_оптимізації.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
