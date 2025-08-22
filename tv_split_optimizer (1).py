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

def heuristic_split_within_group(group_df, total_group_budget):
    """
    Розподіляє фіксований бюджет всередині групи каналів,
    оптимізуючи за вартістю TRP.
    """
    if group_df.empty or total_group_budget == 0:
        group_df['Оптимальна частка (%)'] = 0
        group_df['Оптимальний бюджет'] = 0
        return group_df
    
    # Розрахунок вартості за TRP (як pandas Series для збереження індексів)
    cost_per_trp_series = np.divide(group_df['Ціна'], group_df['TRP'],
                             out=np.full_like(group_df['TRP'].to_numpy(), np.inf, dtype=float),
                             where=group_df['TRP']!=0)
    
    # Сортування індексів від найдешевшого до найдорожчого
    sorted_indices = cost_per_trp_series.sort_values().index
    
    # Розподіл бюджету
    shares = pd.Series(0.0, index=group_df.index)
    remaining_budget = total_group_budget

    for idx in sorted_indices:
        # Доступ до даних за індексом з оригінального DataFrame
        row = group_df.loc[idx]
        cost = cost_per_trp_series.loc[idx]
        
        if pd.notna(cost) and cost != np.inf:
            budget_to_add = min(remaining_budget, row['Ціна'] * row['TRP'])
            shares.loc[idx] = budget_to_add
            remaining_budget -= budget_to_add
        
        if remaining_budget <= 0:
            break

    group_df['Оптимальний бюджет'] = shares
    total_sx_budget = (group_df['Ціна'] * group_df['TRP']).sum() 
    
    # Перераховуємо відсотки
    group_df['Оптимальна частка (%)'] = (group_df['Оптимальний бюджет'] / total_sx_budget) * 100
    
    return group_df


def run_multi_group_optimization(df, buying_audiences, top_channel_groups):
    df['Ціна'] = df.apply(lambda row: row.get(f'Ціна_{buying_audiences.get(row["СХ"], "")}', 0), axis=1)
    df['TRP'] = df.apply(lambda row: row.get(f'TRP_{buying_audiences.get(row["СХ"], "")}', 0), axis=1)
    
    all_results = pd.DataFrame()

    for sh, group_df in df.groupby('СХ'):
        optimized_group = pd.DataFrame()
        total_sx_budget = (group_df['Ціна'] * group_df['TRP']).sum()

        # 1. Оптимізуємо кожну групу "Топ-каналів" окремо
        all_top_channels = [channel for sublist in top_channel_groups.values() for channel in sublist]
        remaining_df = group_df.copy()

        for group_name, channels_list in top_channel_groups.items():
            df_group = group_df[group_df['Канал'].isin(channels_list)].copy()
            if not df_group.empty:
                total_group_budget = (df_group['Ціна'] * df_group['TRP']).sum()
                results_group = heuristic_split_within_group(df_group, total_group_budget)
                optimized_group = pd.concat([optimized_group, results_group])
                # Видаляємо вже оптимізовані канали з решти DataFrame
                remaining_df = remaining_df[~remaining_df['Канал'].isin(channels_list)]

        # 2. Оптимізуємо решту каналів, які не є "топовими"
        if not remaining_df.empty:
            total_remaining_budget = (remaining_df['Ціна'] * remaining_df['TRP']).sum()
            results_remaining = heuristic_split_within_group(remaining_df, total_remaining_budget)
            optimized_group = pd.concat([optimized_group, results_remaining])

        # 3. Перераховуємо відсотки, щоб сума була 100%
        optimized_group['Оптимальна частка (%)'] = (optimized_group['Оптимальний бюджет'] / total_sx_budget) * 100

        all_results = pd.concat([all_results, optimized_group])

    # Фінальна перевірка та нормалізація
    all_results['Оптимальна частка (%)'] = all_results.groupby('СХ')['Оптимальна частка (%)'].transform(lambda x: x / x.sum() * 100)
    
    return all_results.sort_values(['СХ', 'Канал'])

def highlight_cost(val, costs):
    if val == costs.min():
        return 'background-color: lightgreen'
    elif val == costs.max():
        return 'background-color: salmon'
    else:
        return ''

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
    
    st.subheader("📊 Групування каналів")
    st.markdown("**Сумарний бюджет зберігатиметься для кожної групи окремо.**")
    
    # Визначення груп "Топ-каналів" на основі ваших даних
    top_channel_groups = {
        'Оушен': ['СТБ', 'Новий канал', 'ICTV2'],
        'Sirius': ['1+1 Україна', 'ТЕТ', '2+2'],
        'Space': ['НТН']
    }
    
    if st.button("🚀 Запустити оптимізацію"):
        all_results = run_multi_group_optimization(df.copy(), buying_audiences, top_channel_groups)
        
        st.subheader("📊 Результати оптимізації по СХ")
        for sh in all_results['СХ'].unique():
            st.markdown(f"##### СХ: {sh}")
            sh_df = all_results[all_results['СХ']==sh].copy()
            sh_df_sorted = sh_df.sort_values(by='Оптимальна частка (%)', ascending=False)
            
            st.dataframe(
                sh_df_sorted[['Канал','Ціна','TRP','Оптимальна частка (%)','Оптимальний бюджет']]
                .set_index('Канал')
                .style.applymap(lambda v: highlight_cost(v, sh_df_sorted['Ціна']), subset=['Ціна'])
            )
        
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
        
        # --- Експорт у Excel ---
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            all_results.to_excel(writer, sheet_name='Оптимальний спліт', index=False)
        st.download_button("📥 Завантажити результати Excel", data=output.getvalue(),
                             file_name="результати_оптимізації.xlsx")
