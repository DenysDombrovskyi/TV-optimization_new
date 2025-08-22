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

def heuristic_split_percent_with_limits(group_df):
    """
    Евристичний спліт у відсотках з обмеженням на мін/макс відхилення.
    Всі канали залишаються в спліті.
    """
    standard_trp = group_df['TRP'].to_numpy()
    total_trp = standard_trp.sum()
    standard_share = (standard_trp / total_trp) * 100 if total_trp > 0 else np.zeros_like(standard_trp)

    # Мін/макс на основі відхилень
    min_share = np.maximum(standard_share - group_df['Мінімальне відхилення'].to_numpy(), 0)
    max_share = np.minimum(standard_share + group_df['Максимальне відхилення'].to_numpy(), 100)

    # Початковий спліт = мінімальні частки
    shares = min_share.copy()
    remaining = 100 - shares.sum()

    # Вартість за одиницю TRP
    cost_per_trp = np.divide(group_df['Ціна'].to_numpy(), group_df['TRP'].to_numpy(),
                             out=np.full_like(group_df['TRP'].to_numpy(), np.inf, dtype=float),
                             where=group_df['TRP']!=0)

    # Сортуємо від найдешевшого до дорожчого
    sorted_idx = np.argsort(cost_per_trp)

    # Розподіл залишку в межах максимуму
    while remaining > 0:
        updated = False
        for idx in sorted_idx:
            add = min(max_share[idx] - shares[idx], remaining)
            if add > 0:
                shares[idx] += add
                remaining -= add
                updated = True
            if remaining <= 0:
                break
        if not updated:
            # Якщо більше не можна додати до жодного каналу — розподіляємо залишок пропорційно
            shares += remaining / len(shares)
            remaining = 0

    # Перевірка: сума точно 100%
    shares = shares / shares.sum() * 100

    return pd.Series(shares, index=group_df.index)

def run_heuristic_optimization(df, buying_audiences, deviation_df):
    df['Ціна'] = df.apply(lambda row: row.get(f'Ціна_{buying_audiences.get(row["СХ"], "")}', 0), axis=1)
    df['TRP'] = df.apply(lambda row: row.get(f'TRP_{buying_audiences.get(row["СХ"], "")}', 0), axis=1)

    # Об'єднання з deviation_df
    df = df.merge(deviation_df, on='Канал', how='left').fillna(0)

    all_results = pd.DataFrame()

    for sh, group_df in df.groupby('СХ'):
        shares = heuristic_split_percent_with_limits(group_df)
        group_df['Оптимальна частка (%)'] = shares
        group_df['Оптимальний бюджет'] = shares/100 * (group_df['Ціна']*group_df['TRP']).sum()
        all_results = pd.concat([all_results, group_df])

        # --- Додаткова перевірка ---
        total_share = group_df['Оптимальна частка (%)'].sum()
        if not np.isclose(total_share, 100):
            st.warning(f"⚠️ Сума часток для СХ {sh} не дорівнює 100% ({total_share:.2f}%). Автоматично нормалізовано.")
            all_results.loc[group_df.index, 'Оптимальна частка (%)'] = shares / shares.sum() * 100

    return all_results

# --- Streamlit інтерфейс ---

st.set_page_config(page_title="Оптимізація ТВ спліта", layout="wide")
st.title("📺 Евристична оптимізація ТВ спліта з обмеженнями | Dentsu X")

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
    
    st.subheader("📊 Налаштування відхилень по каналах")
    channels_20_percent = ['Новий канал', 'ICTV2', 'СТБ', '1+1 Україна', 'TET', '2+2', 'НТН']
    deviation_df = df[['Канал']].copy()
    deviation_df['Мінімальне відхилення'] = deviation_df['Канал'].apply(lambda x: 20.0 if x in channels_20_percent else 30.0)
    deviation_df['Максимальне відхилення'] = deviation_df['Канал'].apply(lambda x: 20.0 if x in channels_20_percent else 30.0)
    edited_deviation_df = st.data_editor(deviation_df, num_rows="dynamic")
    
    if st.button("🚀 Запустити оптимізацію"):
        all_results = run_heuristic_optimization(df.copy(), buying_audiences, edited_deviation_df)
        
        st.subheader("📊 Результати оптимізації по СХ")
        for sh in all_results['СХ'].unique():
            st.markdown(f"##### СХ: {sh}")
            sh_df = all_results[all_results['СХ']==sh].copy()
            st.dataframe(sh_df[['Канал','Ціна','TRP','Оптимальна частка (%)','Оптимальний бюджет']].set_index('Канал'))
        
        st.subheader("📊 Графіки сплітів")
        for sh in all_results['СХ'].unique():
            sh_df = all_results[all_results['СХ']==sh]
            fig, ax = plt.subplots(figsize=(10,5))
            ax.bar(sh_df['Канал'], sh_df['Оптимальна частка (%)'], color='skyblue')
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
