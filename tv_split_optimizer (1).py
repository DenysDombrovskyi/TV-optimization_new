import streamlit as st
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import io
import numpy as np

# --- Функції для валідації та оптимізації ---

def validate_excel_file(df_standard):
    required_cols_standard = ['Канал', 'СХ']
    for col in required_cols_standard:
        if col not in df_standard.columns:
            st.error(f"❌ Помилка: В аркуші 'Сп-во' відсутній обов'язковий стовпчик '{col}'.")
            return False
    return True

def run_optimization(df, goal, mode, buying_audiences, deviation_df):
    df['Ціна'] = df.apply(lambda row: row.get(f'Ціна_{buying_audiences.get(row["СХ"], "")}', 0), axis=1)
    df['TRP'] = df.apply(lambda row: row.get(f'TRP_{buying_audiences.get(row["СХ"], "")}', 0), axis=1)
    
    total_trp_for_aff = df['TRP'].sum()
    df['Aff'] = (df['TRP'] / total_trp_for_aff) * 100 if total_trp_for_aff > 0 else 0
    
    df['Стандартний Aff'] = df['Aff']
    df['Стандартний TRP'] = df['TRP']
    df['Стандартні слоти'] = 1
    
    df = df.merge(deviation_df, on='Канал', how='left').fillna(0)
    
    all_results = pd.DataFrame()
    
    if mode == 'per_sh':
        with st.spinner('Проводимо оптимізацію за кожним СХ...'):
            for sales_house, group_df in df.groupby('СХ'):
                group_standard_trp = group_df['Стандартний TRP'].sum()
                if group_standard_trp == 0:
                    st.warning(f"Недостатньо даних для СХ {sales_house}. Пропускаємо...")
                    continue
                
                c = group_df['Ціна'].values
                
                # Обмеження по відхиленнях для кожного каналу
                A_channel_ub = np.diag(group_df['TRP'].values)
                b_channel_ub = (1 + group_df['Максимальне відхилення'] / 100) * group_df['Стандартний TRP']
                
                A_channel_lb = -np.diag(group_df['TRP'].values)
                b_channel_lb = -(1 - group_df['Мінімальне відхилення'] / 100) * group_df['Стандартний TRP']
                
                # Обмеження на загальний рейтинг
                A_goal = -group_df[goal].values.reshape(1, -1)
                b_goal = -np.array([group_df[f'Стандартний {goal}'].sum()])

                A = np.vstack((A_channel_ub, A_channel_lb, A_goal))
                b = np.concatenate((b_channel_ub, b_channel_lb, b_goal))
                
                bounds = [(1, None) for _ in range(len(group_df))]

                result = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
                
                if result.success:
                    slots = result.x.round(0).astype(int)
                    group_df['Оптимальні слоти'] = slots
                    group_df['Оптимальний TRP'] = slots * group_df['TRP']
                    group_df['Оптимальний Aff'] = slots * group_df['Aff']
                    all_results = pd.concat([all_results, group_df])
                else:
                    st.error(f"❌ Оптимізація для СХ {sales_house} не вдалася: {result.message}")
                    
    else:  # mode == 'total'
        with st.spinner('Проводимо загальну оптимізацію...'):
            c = df['Ціна'].values

            A_channel_ub = np.diag(df['TRP'].values)
            b_channel_ub = (1 + df['Максимальне відхилення'] / 100) * df['Стандартний TRP']
            
            A_channel_lb = -np.diag(df['TRP'].values)
            b_channel_lb = -(1 - df['Мінімальне відхилення'] / 100) * df['Стандартний TRP']

            A_goal = -df[goal].values.reshape(1, -1)
            b_goal = -np.array([df[f'Стандартний {goal}'].sum()])

            A = np.vstack((A_channel_ub, A_channel_lb, A_goal))
            b = np.concatenate((b_channel_ub, b_channel_lb, b_goal))
            
            bounds = [(1, None) for _ in range(len(df))]
            result = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
            
            if result.success:
                slots = result.x.round(0).astype(int)
                df['Оптимальні слоти'] = slots
                df['Оптимальний TRP'] = slots * df['TRP']
                df['Оптимальний Aff'] = slots * df['Aff']
                all_results = df
            else:
                st.error(f"❌ Загальна оптимізація не вдалася: {result.message}")
    
    return all_results

# --- Основна частина програми Streamlit ---

st.set_page_config(page_title="Оптимізація ТВ спліта", layout="wide")
st.title("📺 Оптимізація ТВ спліта | Dentsu X")

uploaded_file = st.file_uploader("Завантажте Excel-файл з даними", type=["xlsx"])

if uploaded_file:
    try:
        with st.spinner('Завантаження та обробка даних...'):
            standard_df = pd.read_excel(uploaded_file, sheet_name="Сп-во", skiprows=2, engine="openpyxl")
            
            if not validate_excel_file(standard_df):
                st.stop()
            
            st.success("✅ Дані успішно завантажено!")
            all_data = standard_df.copy()
            
    except Exception as e:
        st.error(f"❌ Сталася неочікувана помилка: {e}")
    else:
        all_sh = all_data['СХ'].unique()
        all_ba = [col.replace('Ціна_', '') for col in all_data.columns if 'Ціна_' in col]
        
        st.header("🔧 Налаштування оптимізації")
        goal = st.selectbox("Мета оптимізації", ['Aff', 'TRP'])
        mode = st.selectbox("Режим оптимізації", ['total', 'per_sh'])
        
        st.subheader("🎯 Вибір БА для кожного СХ")
        buying_audiences = {}
        for sh in all_sh:
            ba = st.selectbox(f"СХ: {sh}", all_ba, key=sh)
            buying_audiences[sh] = ba
        
        st.subheader("📊 Налаштування відхилень по каналах")
        channels_20_percent = ['Новий канал', 'ICTV2', 'СТБ', '1+1 Україна', 'TET', '2+2', 'НТН']
        deviation_df = all_data[['Канал']].copy()
        
        def set_default_deviation(channel):
            return 20.0 if channel in channels_20_percent else 30.0

        deviation_df['Мінімальне відхилення'] = deviation_df['Канал'].apply(set_default_deviation)
        deviation_df['Максимальне відхилення'] = deviation_df['Канал'].apply(set_default_deviation)
        edited_deviation_df = st.data_editor(deviation_df, num_rows="dynamic")
        
        if st.button("🚀 Запустити оптимізацію"):
            all_results = run_optimization(all_data.copy(), goal, mode, buying_audiences, edited_deviation_df)
            
            if not all_results.empty:
                all_results['Оптимальний бюджет'] = all_results['Оптимальні слоти'] * all_results['Ціна']
                all_results['Стандартний бюджет'] = all_results['Стандартні слоти'] * all_results['Ціна']
                
                total_budget_opt = all_results['Оптимальний бюджет'].sum()
                total_budget_std = all_results['Стандартний бюджет'].sum()
                
                total_trp_opt = all_results['Оптимальний TRP'].sum()
                total_aff_opt = all_results['Оптимальний Aff'].sum()
                
                total_trp_std = all_results['Стандартний TRP'].sum()
                total_aff_std = all_results['Стандартний Aff'].sum()
                
                cpp_opt = total_budget_opt / total_trp_opt if total_trp_opt > 0 else 0
                cpp_std = total_budget_std / total_trp_std if total_trp_std > 0 else 0
                cpt_opt = total_budget_opt / total_aff_opt if total_aff_opt > 0 else 0
                cpt_std = total_budget_std / total_aff_std if total_aff_std > 0 else 0
                
                if total_budget_opt == total_budget_std:
                    st.info("ℹ️ Оптимізація не знайшла можливості зменшити вартість.")
                else:
                    st.success(f"✅ Оптимізація завершена! Економія бюджету: {(total_budget_std - total_budget_opt):,.2f} грн")
                
                st.subheader("📊 Результати оптимізації")
                tab1, tab2, tab3, tab4 = st.tabs(["Загальні показники", "Деталі по СХ", "Графіки", "Спліт по СХ"])

                # --- Загальні показники ---
                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info("**Стандартний спліт**")
                        st.metric("Ціна за Aff", f"{cpt_std:,.2f} грн")
                        st.metric("Ціна за TRP", f"{cpp_std:,.2f} грн")
                        st.metric("Загальний бюджет", f"{total_budget_std:,.2f} грн")
                    with col2:
                        st.success("**Оптимізований спліт**")
                        st.metric("Ціна за Aff", f"{cpt_opt:,.2f} грн")
                        st.metric("Ціна за TRP", f"{cpp_opt:,.2f} грн")
                        st.metric("Загальний бюджет", f"{total_budget_opt:,.2f} грн")

                # --- Оптимальний спліт по СХ ---
                with tab4:
                    st.markdown("#### Оптимальний спліт по кожному СХ")
                    for sh in all_results['СХ'].unique():
                        st.markdown(f"##### СХ: {sh}")
                        sh_df = all_results[all_results['СХ'] == sh].copy()
                        sh_df['Стандартна частка TRP'] = (sh_df['Стандартний TRP'] / sh_df['Стандартний TRP'].sum()) * 100
                        sh_df['Оптимальна частка TRP'] = (sh_df['Оптимальний TRP'] / sh_df['Оптимальний TRP'].sum()) * 100
                        sh_df['Стандартна частка бюджету'] = (sh_df['Стандартний бюджет'] / sh_df['Стандартний бюджет'].sum()) * 100
                        sh_df['Оптимальна частка бюджету'] = (sh_df['Оптимальний бюджет'] / sh_df['Оптимальний бюджет'].sum()) * 100
                        st.dataframe(sh_df[['Канал', 'Стандартні слоти', 'Стандартний TRP', 'Стандартний Aff', 
                                            'Оптимальні слоти', 'Оптимальний TRP', 'Оптимальний Aff', 
                                            'Стандартна частка TRP', 'Оптимальна частка TRP', 
                                            'Стандартна частка бюджету', 'Оптимальна частка бюджету']].set_index('Канал'))                
                    # Графік 2: Загальний TRP
                    fig_trp_abs, ax_trp_abs = plt.subplots(figsize=(12, 6))
                    width = 0.35
                    x = range(len(labels))
                    ax_trp_abs.bar(x, all_results['Стандартний TRP'], width, label='Стандартний спліт', color='gray')
                    ax_trp_abs.bar([p + width for p in x], all_results['Оптимальний TRP'], width, label='Оптимізований спліт', color='skyblue')
                    
                    ax_trp_abs.set_title('Загальний TRP')
                    ax_trp_abs.set_ylabel('TRP')
                    ax_trp_abs.set_xticks([p + width / 2 for p in x])
                    ax_trp_abs.set_xticklabels(labels, rotation=45, ha="right")
                    ax_trp_abs.legend()
                    ax_trp_abs.grid(axis='y')
                    plt.tight_layout()
                    st.pyplot(fig_trp_abs)

                    # Графік 3: Кількість слотів
                    fig_slots, ax_slots = plt.subplots(figsize=(12, 6))
                    ax_slots.bar(all_results['Канал'], all_results['Оптимальні слоти'], color='skyblue')
                    
                    ax_slots.set_title('Кількість слотів в оптимізованому спліті')
                    ax_slots.set_ylabel('Кількість слотів')
                    ax_slots.set_xlabel('Канал')
                    ax_slots.set_xticklabels(all_results['Канал'], rotation=45, ha="right")
                    ax_slots.grid(axis='y')
                    plt.tight_layout()
                    st.pyplot(fig_slots)
                    
                # Кнопка для завантаження результатів
                st.markdown("---")
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    
                    excel_df = all_results[['Канал', 'СХ', 
                                            'Стандартні слоти', 'Стандартний TRP', 'Стандартний Aff', 'Стандартний бюджет',
                                            'Оптимальні слоти', 'Оптимальний TRP', 'Оптимальний Aff', 'Оптимальний бюджет']].copy()
                    
                    total_row = pd.DataFrame([['Загалом', '-', 
                                               excel_df['Стандартні слоти'].sum(), 
                                               excel_df['Стандартний TRP'].sum(), 
                                               excel_df['Стандартний Aff'].sum(), 
                                               excel_df['Стандартний бюджет'].sum(),
                                               excel_df['Оптимальні слоти'].sum(), 
                                               excel_df['Оптимальний TRP'].sum(), 
                                               excel_df['Оптимальний Aff'].sum(), 
                                               excel_df['Оптимальний бюджет'].sum()]],
                                              columns=excel_df.columns)
                    excel_df = pd.concat([excel_df, total_row], ignore_index=True)
                    excel_df.to_excel(writer, sheet_name='Спліт', index=False)

                    display_df_sh_costs.set_index('СХ').to_excel(writer, sheet_name='Вартість по СХ')

                st.download_button("📥 Завантажити результати Excel", data=output.getvalue(), file_name="результати_оптимізації.xlsx")
