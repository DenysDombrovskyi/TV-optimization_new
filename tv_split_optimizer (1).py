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

def heuristic_split_within_group(group_df, total_group_budget, min_share, max_share):
    """
    Розподіляє фіксований бюджет всередині групи каналів з обмеженням часток,
    оптимізуючи за TRP.
    """
    if group_df.empty or total_group_budget == 0:
        group_df['Оптимальна частка (%)'] = 0
        group_df['Оптимальний бюджет'] = 0
        return group_df

    # Ініціалізація
    shares = pd.Series(0.0, index=group_df.index)
    remaining_budget = total_group_budget

    # Попередній розподіл з урахуванням обмежень
    for idx, row in group_df.iterrows():
        min_b = min_share.get(row['Канал'], 0) / 100 * total_group_budget
        max_b = max_share.get(row['Канал'], 1) / 100 * total_group_budget
        alloc = min(max_b, remaining_budget)
        shares.loc[idx] = alloc
        remaining_budget -= alloc

    # Якщо залишився бюджет, розподіляємо пропорційно TRP*Affinity, не перевищуючи max
    if remaining_budget > 0:
        cost_eff = row['Affinity'] / group_df['Affinity'].sum()
        additional = remaining_budget * cost_eff
        shares += additional
        shares = shares.clip(upper=[max_share.get(k, 1)/100*total_group_budget for k in group_df['Канал']])

    group_df['Оптимальний бюджет'] = shares
    total_sx_budget = (group_df['Ціна'] * group_df['TRP']).sum()
    group_df['Оптимальна частка (%)'] = (group_df['Оптимальний бюджет'] / total_sx_budget) * 100
    return group_df

def run_multi_group_optimization(df, buying_audiences, top_channel_groups, min_share, max_share):
    # Підготовка колонок
    df['Ціна'] = df.apply(lambda row: row.get(f'Ціна_{buying_audiences.get(row["СХ"], "")}', 0), axis=1)
    df['Affinity'] = df.apply(lambda row: row.get(f'Affinity_{buying_audiences.get(row["СХ"], "")}', 1.0), axis=1)
    
    # Якщо TRP немає — розраховуємо як частку бюджету помножити на Affinity
    for sh in df['СХ'].unique():
        df_sh = df[df['СХ']==sh]
        total_price = df_sh['Ціна'].sum()
        df.loc[df['СХ']==sh, 'TRP'] = (df_sh['Ціна']/total_price) * df_sh['Affinity']
    
    all_results = pd.DataFrame()

    for sh, group_df in df.groupby('СХ'):
        optimized_group = pd.DataFrame()
        remaining_df_for_opt = group_df.copy()
        
        # Обробляємо Топ-канали
        for group_name, channels_list in top_channel_groups.items():
            df_group = remaining_df_for_opt[remaining_df_for_opt['Канал'].isin(channels_list)].copy()
            if not df_group.empty:
                total_group_budget = (df_group['Ціна'] * df_group['TRP']).sum()
                results_group = heuristic_split_within_group(df_group, total_group_budget, min_share, max_share)
                optimized_group = pd.concat([optimized_group, results_group])
                remaining_df_for_opt = remaining_df_for_opt[~remaining_df_for_opt['Канал'].isin(channels_list)]
        
        # Обробляємо решту каналів
        if not remaining_df_for_opt.empty:
            total_remaining_budget = (remaining_df_for_opt['Ціна'] * remaining_df_for_opt['TRP']).sum()
            results_remaining = heuristic_split_within_group(remaining_df_for_opt, total_remaining_budget, min_share, max_share)
            optimized_group = pd.concat([optimized_group, results_remaining])
        
        # Перерахунок відсотків
        total_budget_group = optimized_group['Оптимальний бюджет'].sum()
        optimized_group['Оптимальна частка (%)'] = optimized_group['Оптимальний бюджет'] / total_budget_group * 100

        # Розрахунок GRP та TRP
        optimized_group['GRP'] = optimized_group['Оптимальний бюджет'] / optimized_group['Ціна']
        optimized_group['TRP'] = optimized_group['GRP'] * optimized_group['Affinity']
        
        all_results = pd.concat([all_results, optimized_group])
    
    return all_results.sort_values(['СХ', 'Канал'])

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
        df = pd.read_excel(uploaded_file, sheet_name="Сп-во", skiprows=2, engine="openpyxl")
        if not validate_excel_file(df):
            st.stop()
        st.success("✅ Дані успішно завантажено!")
    except Exception as e:
        st.error(f"❌ Помилка при завантаженні файлу: {e}")
        st.stop()
    
    all_sh = df['СХ'].unique()
    all_ba = [col.replace('Ціна_', '') for col in df.columns if 'Ціна_' in col]
    
    st.header("🔧 Налаштування оптимізації")
    st.subheader("🎯 Вибір БА для кожного СХ")
    buying_audiences = {}
    for sh in all_sh:
        ba = st.selectbox(f"СХ: {sh}", all_ba, key=sh)
        buying_audiences[sh] = ba
    
    st.subheader("📊 Групування каналів та обмеження часток")
    st.markdown("Можна задати мінімальну і максимальну частку бюджету (%) для кожного каналу:")
    min_share = {}
    max_share = {}
    for channel in df['Канал'].unique():
        col1, col2 = st.columns(2)
        with col1:
            min_val = st.number_input(f"Min % {channel}", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"min_{channel}")
            min_share[channel] = min_val
        with col2:
            max_val = st.number_input(f"Max % {channel}", min_value=0.0, max_value=100.0, value=100.0, step=1.0, key=f"max_{channel}")
            max_share[channel] = max_val
    
    # Топ-канали
    top_channel_groups = {
        'Оушен': ['СТБ', 'Новий канал', 'ICTV2'],
        'Sirius': ['1+1 Україна', 'ТЕТ', '2+2'],
        'Space': ['НТН']
    }
    
    if st.button("🚀 Запустити оптимізацію"):
        all_results = run_multi_group_optimization(df.copy(), buying_audiences, top_channel_groups, min_share, max_share)
        all_top_channels = [channel for sublist in top_channel_groups.values() for channel in sublist]
        
        st.subheader("📊 Результати оптимізації по СХ")
        for sh in all_results['СХ'].unique():
            st.markdown(f"##### СХ: {sh}")
            sh_df = all_results[all_results['СХ']==sh].copy()
            sh_df_sorted = sh_df.sort_values(by='Оптимальна частка (%)', ascending=False)
            sh_df_sorted['Початковий бюджет'] = sh_df_sorted['TRP'] * sh_df_sorted['Ціна']
            
            st.dataframe(
                sh_df_sorted[['Канал', 'Початковий бюджет', 'Оптимальний бюджет', 'GRP', 'TRP', 'Оптимальна частка (%)']]
                .style.apply(highlight_top_channels, axis=1, top_channels=all_top_channels)
            )

            st.markdown(f"**Сумарний початковий бюджет:** `{sh_df_sorted['Початковий бюджет'].sum():,.2f}`")
            st.markdown(f"**Сумарний оптимальний бюджет:** `{sh_df_sorted['Оптимальний бюджет'].sum():,.2f}`")
            st.markdown(f"**Сумарний бюджет Топ-каналів:** `{sh_df_sorted[sh_df_sorted['Канал'].isin(all_top_channels)]['Оптимальний бюджет'].sum():,.2f}`")
        
        st.subheader("📊 Графіки сплітів")
        for sh in all_results['СХ'].unique():
            sh_df = all_results[all_results['СХ']==sh]
            fig, ax = plt.subplots(figsize=(10,5))
            colors = ['lightgreen' if c==sh_df['Ціна'].min() else 'salmon' if c==sh_df['Ціна'].max() else 'skyblue' for c in sh_df['Ціна']]
            ax.bar(sh_df['Канал'], sh_df['Оптимальна частка (%)'], color=colors)
            ax.set_ylabel('Частка (%)')
            ax.set_title(f"СХ: {sh} — Оптимальна частка по каналах")
            ax.set_xticklabels(sh_df['Канал'], rotation=45, ha='right')
            ax.grid(axis='y')
            st.pyplot(fig)
        
        # Експорт Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            all_results.to_excel(writer, sheet_name='Оптимальний спліт', index=False)
        st.download_button("📥 Завантажити результати Excel", data=output.getvalue(),
                             file_name="результати_оптимізації.xlsx")
