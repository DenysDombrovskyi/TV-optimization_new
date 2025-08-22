import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

# --- Функції ---
def validate_excel_file(df_standard):
    required_cols_standard = ['Канал', 'СХ']
    for col in required_cols_standard:
        if col not in df_standard.columns:
            st.error(f"❌ В аркуші 'Сп-во' відсутній обов'язковий стовпчик '{col}'.")
            return False
    return True

def apply_budget_limits(df, min_share, max_share):
    df = df.copy()
    for idx, row in df.iterrows():
        channel = row['Канал']
        base = row['Бюджет_оптимальний']
        min_val = min_share.get(channel, 0)
        max_val = max_share.get(channel, 100)
        df.at[idx, 'Оптимальний бюджет'] = np.clip(base, min_val, max_val)
    df['Оптимальний бюджет'] = df['Оптимальний бюджет'] / df['Оптимальний бюджет'].sum() * 100
    return df

def calculate_grp_trp(df):
    df = df.copy()
    df['GRP'] = df['Оптимальний бюджет'] / df['Ціна_оптимальна']
    df['TRP'] = df['GRP'] * df['Affinity']
    return df

def highlight_top_channels(row, top_channels):
    is_top_channel = row['Канал'] in top_channels
    style = 'font-weight: bold; background-color: #f0f0f0' if is_top_channel else ''
    return [style] * len(row)

# --- Streamlit інтерфейс ---
st.set_page_config(page_title="Оптимізація ТВ спліта", layout="wide")
st.title("📺 Евристична оптимізація ТВ спліта | Dentsu X")

uploaded_file = st.file_uploader("Завантажте Excel-файл з даними", type=["xlsx"])

if uploaded_file:
    try:
        # Основний лист і лист з Affinity
        df_main = pd.read_excel(uploaded_file, sheet_name="Сп-во", skiprows=2, engine="openpyxl")
        df_affinity = pd.read_excel(uploaded_file, sheet_name="Affinity", engine="openpyxl")
        
        if not validate_excel_file(df_main):
            st.stop()
        st.success("✅ Дані успішно завантажено!")
        
        # З'єднуємо по Каналу
        df = df_main.merge(df_affinity, on='Канал', how='left')
        df['Affinity'].fillna(1.0, inplace=True)  # якщо немає Affinity, ставимо 1.0
        
    except Exception as e:
        st.error(f"❌ Помилка при завантаженні файлу: {e}")
        st.stop()

    all_sh = df['СХ'].unique()

    st.header("🔧 Налаштування оптимізації")
    st.subheader("🎯 Вибір БА для кожного СХ")
    buying_audiences = {}
    for sh in all_sh:
        ba_options = [col.replace('Ціна_', '') for col in df.columns if col.startswith('Ціна_')]
        ba = st.selectbox(f"СХ: {sh}", ba_options, key=sh)
        buying_audiences[sh] = ba

    # Топ-канали
    top_channel_groups = {
        'Оушен': ['СТБ', 'Новий канал', 'ICTV2'],
        'Sirius': ['1+1 Україна', 'ТЕТ', '2+2'],
        'Space': ['НТН']
    }
    all_top_channels = [ch for sublist in top_channel_groups.values() for ch in sublist]

    # Мін/макс частки
    min_share = {}
    max_share = {}
    for channel in df['Канал'].unique():
        if channel in all_top_channels:
            min_val = 80.0
            max_val = 120.0
        else:
            min_val = 70.0
            max_val = 130.0
        min_share[channel] = min_val
        max_share[channel] = max_val

    # Підготовка базових колонок бюджету і ціни
    df['Бюджет_оптимальний'] = df.apply(
        lambda row: row.get(f'Бюджет_{buying_audiences.get(row["СХ"], "")}', 
                            row.get('Бюджет (%)', 1.0)),  # дефолт 1.0
        axis=1
    )
    df['Ціна_оптимальна'] = df.apply(
        lambda row: row.get(f'Ціна_{buying_audiences.get(row["СХ"], "")}', 1.0),  # дефолт 1.0
        axis=1
    )

    if st.button("🚀 Запустити оптимізацію"):
        all_results = pd.DataFrame()
        for sh in all_sh:
            df_sh = df[df['СХ']==sh].copy()
            df_sh = apply_budget_limits(df_sh, min_share, max_share)
            df_sh = calculate_grp_trp(df_sh)
            
            # Сумарна частка бюджету топ-каналів по СХ
            mask_top = df_sh['Канал'].isin(all_top_channels)
            sum_top_budget = df_sh.loc[mask_top, 'Оптимальний бюджет'].sum()
            df_sh['Сумарна частка бюджету топ-каналів (%)'] = sum_top_budget
            
            all_results = pd.concat([all_results, df_sh])

        st.subheader("📊 Результати оптимізації по СХ")
        for sh in all_results['СХ'].unique():
            st.markdown(f"##### СХ: {sh}")
            sh_df = all_results[all_results['СХ']==sh].copy()
            st.dataframe(
                sh_df[['Канал', 'Бюджет_оптимальний', 'Оптимальний бюджет', 'Ціна_оптимальна', 'GRP', 'TRP', 'Сумарна частка бюджету топ-каналів (%)']]
                .style.apply(highlight_top_channels, axis=1, top_channels=all_top_channels)
            )

            st.markdown(f"**Сумарний оптимальний бюджет:** `{sh_df['Оптимальний бюджет'].sum():,.2f}`")
            st.markdown(f"**Сумарний бюджет топ-каналів:** `{sh_df.loc[sh_df['Канал'].isin(all_top_channels), 'Оптимальний бюджет'].sum():,.2f}`")

        st.subheader("📊 Графіки сплітів")
        for sh in all_results['СХ'].unique():
            sh_df = all_results[all_results['СХ']==sh]
            fig, ax = plt.subplots(figsize=(10,5))
            colors = ['lightgreen' if c==sh_df['Ціна_оптимальна'].min() else 'salmon' if c==sh_df['Ціна_оптимальна'].max() else 'skyblue' for c in sh_df['Ціна_оптимальна']]
            ax.bar(sh_df['Канал'], sh_df['Оптимальний бюджет'], color=colors)
            ax.set_ylabel('Бюджет (%)')
            ax.set_title(f"СХ: {sh} — Оптимальний спліт по каналах")
            ax.set_xticklabels(sh_df['Канал'], rotation=45, ha='right')
            ax.grid(axis='y')
            st.pyplot(fig)

        # --- Експорт у Excel ---
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            all_results.to_excel(writer, sheet_name='Оптимальний спліт', index=False)
        st.download_button("📥 Завантажити результати Excel", data=output.getvalue(),
                             file_name="результати_оптимізації.xlsx")
