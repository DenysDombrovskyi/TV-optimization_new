import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

# --- Функції для валідації та оптимізації ---

def validate_excel_file(df_standard):
    required_cols_standard = ['Канал', 'СХ']
    for col in required_cols_standard:
        if col not in df_standard.columns:
            st.error(f"❌ Помилка: В аркуші 'Сп-во' відсутній обов'язковий стовпчик '{col}'.")
            return False
    return True

def heuristic_split(group_df):
    """
    Евристичний розподіл слотів для всіх каналів з дотриманням мін/макс відхилень.
    """
    min_slots = np.floor(group_df['Стандартні слоти'] * (1 - group_df['Мінімальне відхилення']/100)).astype(int).to_numpy()
    max_slots = np.ceil(group_df['Стандартні слоти'] * (1 + group_df['Максимальне відхилення']/100)).astype(int).to_numpy()
    
    total_slots = group_df['Стандартні слоти'].sum()
    slots = np.round(group_df['TRP'] / group_df['TRP'].sum() * total_slots).astype(int)
    slots = np.array(slots)  # NumPy масив для змін
    
    slots = np.clip(slots, min_slots, max_slots)
    
    diff = total_slots - slots.sum()
    trp_values = group_df['TRP'].to_numpy()
    
    while diff != 0:
        if diff > 0:
            candidates = np.where(slots < max_slots)[0]
            if len(candidates) == 0:
                break
            eff_idx = candidates[np.argmax(trp_values[candidates])]
            slots[eff_idx] += 1
            diff -= 1
        else:
            candidates = np.where(slots > min_slots)[0]
            if len(candidates) == 0:
                break
            eff_idx = candidates[np.argmin(trp_values[candidates])]
            slots[eff_idx] -= 1
            diff += 1
    
    return pd.Series(slots, index=group_df.index)

def run_heuristic_optimization(df, goal, buying_audiences, deviation_df):
    df['Ціна'] = df.apply(lambda row: row.get(f'Ціна_{buying_audiences.get(row["СХ"], "")}', 0), axis=1)
    df['TRP'] = df.apply(lambda row: row.get(f'TRP_{buying_audiences.get(row["СХ"], "")}', 0), axis=1)
    
    total_trp_for_aff = df['TRP'].sum()
    df['Aff'] = (df['TRP'] / total_trp_for_aff) * 100 if total_trp_for_aff > 0 else 0
    
    df['Стандартний Aff'] = df['Aff']
    df['Стандартний TRP'] = df['TRP']
    df['Стандартні слоти'] = 1
    
    df = df.merge(deviation_df, on='Канал', how='left').fillna(0)
    
    all_results = pd.DataFrame()
    
    for sh, group_df in df.groupby('СХ'):
        slots = heuristic_split(group_df)
        group_df['Оптимальні слоти'] = slots
        group_df['Оптимальний TRP'] = slots * group_df['TRP']
        group_df['Оптимальний Aff'] = slots * group_df['Aff']
        group_df['Оптимальний бюджет'] = slots * group_df['Ціна']
        group_df['Стандартний бюджет'] = group_df['Стандартні слоти'] * group_df['Ціна']
        all_results = pd.concat([all_results, group_df])
    
    return all_results

# --- Streamlit інтерфейс ---

st.set_page_config(page_title="Оптимізація ТВ спліта", layout="wide")
st.title("📺 Оптимізація ТВ спліта | Dentsu X")

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
    goal = st.selectbox("Мета оптимізації", ['Aff', 'TRP'])
    
    st.subheader("🎯 Вибір БА для кожного СХ")
    buying_audiences = {}
    for sh in all_sh:
        ba = st.selectbox(f"СХ: {sh}", all_ba, key=sh)
        buying_audiences[sh] = ba
    
    st.subheader("📊 Налаштування відхилень по каналах")
    channels_20_percent = ['Новий канал', 'ICTV2', 'СТБ', '1+1 Україна', 'TET', '2+2', 'НТН']
    deviation_df = df[['Канал']].copy()
    deviation_df['Мінімальне відхилення'] = deviation_df['Канал'].apply(lambda x: 20.0 if x in channels_20_percent else 30.0)
    deviation_df['Максимальне відхилення'] = deviation_df['Канал'].apply(lambda x: 20.0 if x in channels_20_percent else 30.0)
    edited_deviation_df = st.data_editor(deviation_df, num_rows="dynamic")
    
    if st.button("🚀 Запустити оптимізацію"):
        all_results = run_heuristic_optimization(df.copy(), goal, buying_audiences, edited_deviation_df)
        
        st.subheader("📊 Результати оптимізації по СХ")
        for sh in all_results['СХ'].unique():
            st.markdown(f"##### СХ: {sh}")
            sh_df = all_results[all_results['СХ']==sh].copy()
            sh_df['Стандартна частка TRP'] = (sh_df['Стандартний TRP']/sh_df['Стандартний TRP'].sum())*100
            sh_df['Оптимальна частка TRP'] = (sh_df['Оптимальний TRP']/sh_df['Оптимальний TRP'].sum())*100
            sh_df['Стандартна частка бюджету'] = (sh_df['Стандартний бюджет']/sh_df['Стандартний бюджет'].sum())*100
            sh_df['Оптимальна частка бюджету'] = (sh_df['Оптимальний бюджет']/sh_df['Оптимальний бюджет'].sum())*100
            st.dataframe(sh_df[['Канал','Стандартні слоти','Стандартний TRP','Стандартний Aff',
                                'Оптимальні слоти','Оптимальний TRP','Оптимальний Aff',
                                'Стандартна частка TRP','Оптимальна частка TRP',
                                'Стандартна частка бюджету','Оптимальна частка бюджету']].set_index('Канал'))
        
        st.subheader("📊 Графіки сплітів")
        for sh in all_results['СХ'].unique():
            sh_df = all_results[all_results['СХ']==sh]
            fig, ax = plt.subplots(figsize=(10,5))
            width = 0.35
            x = np.arange(len(sh_df))
            ax.bar(x - width/2, sh_df['Стандартний бюджет'], width, label='Стандартний', color='gray')
            ax.bar(x + width/2, sh_df['Оптимальний бюджет'], width, label='Оптимальний', color='skyblue')
            ax.set_xticks(x)
            ax.set_xticklabels(sh_df['Канал'], rotation=45, ha='right')
            ax.set_ylabel('Бюджет')
            ax.set_title(f"СХ: {sh} — Розподіл бюджету по каналах")
            ax.legend()
            ax.grid(axis='y')
            st.pyplot(fig)
        
        # --- Експорт у Excel ---
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            all_results.to_excel(writer, sheet_name='Оптимальний спліт', index=False)
        st.download_button("📥 Завантажити результати Excel", data=output.getvalue(),
                           file_name="результати_оптимізації.xlsx")
