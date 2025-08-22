import streamlit as st
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import io

# --- Функції для валідації та оптимізації ---

def validate_excel_file(df_standard):
    """
    Перевіряє, чи існують необхідні стовпці в завантажених датафреймах.
    Повертає True, якщо валідація успішна, інакше - False.
    """
    required_cols_standard = ['Канал', 'СХ']

    for col in required_cols_standard:
        if col not in df_standard.columns:
            st.error(f"❌ Помилка: В аркуші 'Сп-во' відсутній обов'язковий стовпчик '{col}'.")
            return False
    
    return True

def run_optimization(df, goal, mode, buying_audiences, deviation_df):
    """
    Виконує оптимізацію ТВ-спліта на основі даних і налаштувань.
    Повертає датафрейм з результатами оптимізації.
    """
    # Додаємо стовпці з цінами та TRP для обраної БА
    df['Ціна'] = df.apply(lambda row: row.get(f'Ціна_{buying_audiences.get(row["СХ"], "")}', 0), axis=1)
    df['TRP'] = df.apply(lambda row: row.get(f'TRP_{buying_audiences.get(row["СХ"], "")}', 0), axis=1)
    
    # НОВА ФОРМУЛА ДЛЯ AFFINITY ПО КАНАЛАХ
    total_trp_for_aff = df['TRP'].sum()
    df['Aff'] = (df['TRP'] / total_trp_for_aff) * 100 if total_trp_for_aff > 0 else 0
    
    df['Стандартний Aff'] = df['Aff']
    df['Стандартний TRP'] = df['TRP']
    df['Стандартні слоти'] = 1  
    
    # Об'єднуємо дані з таблиці відхилень
    df = df.merge(deviation_df, on='Канал', how='left').fillna(0)
    
    all_results = pd.DataFrame()
    
    if mode == 'per_sh':
        with st.spinner('Проводимо оптимізацію за кожним СХ...'):
            for sales_house, group_df in df.groupby('СХ'):
                group_standard_trp = group_df['Стандартний TRP'].sum()
                if group_standard_trp == 0:
                    st.warning(f"Недостатньо даних для СХ {sales_house}. Пропускаємо...")
                    continue
                
                group_df['Стандартна доля TRP'] = group_df['Стандартний TRP'] / group_standard_trp
                
                # Завдання для linprog
                c = group_df['Ціна'].values # Мета - мінімізувати вартість

                # Обмеження на частку TRP по кожному каналу
                A_upper = pd.get_dummies(group_df['Канал']).mul(group_df['TRP'], axis=0).values
                b_upper = (1 + group_df['Максимальне відхилення'] / 100) * group_df['Стандартний TRP']
                
                A_lower = -pd.get_dummies(group_df['Канал']).mul(group_df['TRP'], axis=0).values
                b_lower = -(1 - group_df['Мінімальне відхилення'] / 100) * group_df['Стандартний TRP']

                A = list(A_upper) + list(A_lower)
                b = list(b_upper) + list(b_lower)

                # Додаємо нове обмеження: загальний рейтинг (TRP або Aff) має бути не менше стандартного
                A_ub_goal = [-group_df[goal].values]
                b_ub_goal = [-group_df[f'Стандартний {goal}'].sum()]

                A.extend(A_ub_goal)
                b.extend(b_ub_goal)
                
                # Кожен канал повинен мати хоча б 1 слот
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
            total_standard_trp = df['Стандартний TRP'].sum()
            
            # Завдання для linprog
            c = df['Ціна'].values # Мета - мінімізувати вартість

            # Обмеження на частку TRP по кожному каналу
            A_upper = pd.get_dummies(df['Канал']).mul(df['TRP'], axis=0).values
            b_upper = (1 + df['Максимальне відхилення'] / 100) * df['Стандартний TRP']
            
            A_lower = -pd.get_dummies(df['Канал']).mul(df['TRP'], axis=0).values
            b_lower = -(1 - df['Мінімальне відхилення'] / 100) * df['Стандартний TRP']

            A = list(A_upper) + list(A_lower)
            b = list(b_upper) + list(b_lower)
            
            # Додаємо нове обмеження: загальний рейтинг (TRP або Aff) має бути не менше стандартного
            A_ub_goal = [-df[goal].values]
            b_ub_goal = [-df[f'Стандартний {goal}'].sum()]

            A.extend(A_ub_goal)
            b.extend(b_ub_goal)

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
            
    except FileNotFoundError:
        st.error("❌ Помилка: Не знайдено необхідний аркуш в файлі. Переконайтеся, що файл містить аркуш 'Сп-во'.")
    except KeyError as e:
        st.error(f"❌ Помилка: У файлі відсутній необхідний стовпець: {e}.")
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
            if channel in channels_20_percent:
                return 20.0
            return 30.0

        deviation_df['Мінімальне відхилення'] = deviation_df['Канал'].apply(set_default_deviation)
        deviation_df['Максимальне відхилення'] = deviation_df['Канал'].apply(set_default_deviation)
        
        edited_deviation_df = st.data_editor(deviation_df, num_rows="dynamic")
        
        if st.button("🚀 Запустити оптимізацію"):
            all_results = run_optimization(all_data.copy(), goal, mode, buying_audiences, edited_deviation_df)
            
            if not all_results.empty:
                # Розрахунок бюджетів
                all_results['Оптимальний бюджет'] = all_results['Оптимальні слоти'] * all_results['Ціна']
                all_results['Стандартний бюджет'] = all_results['Стандартні слоти'] * all_results['Ціна']
                
                # Розрахунок показників для порівняння
                total_budget_opt = all_results['Оптимальний бюджет'].sum()
                total_budget_std = all_results['Стандартний бюджет'].sum()
                
                total_trp_opt = all_results['Оптимальний TRP'].sum()
                total_aff_opt = all_results['Оптимальний Aff'].sum()
                
                total_trp_std = all_results['Стандартний TRP'].sum()
                total_aff_std = all_results['Стандартний Aff'].sum()
                
                # Розрахунок вартості за рейтинг для загального спліта
                cpp_opt = total_budget_opt / total_trp_opt if total_trp_opt > 0 else 0
                cpp_std = total_budget_std / total_trp_std if total_trp_std > 0 else 0
                cpt_opt = total_budget_opt / total_aff_opt if total_aff_opt > 0 else 0
                cpt_std = total_budget_std / total_aff_std if total_aff_std > 0 else 0
                
                # Порівняння результатів
                if (all_results['Оптимальні слоти'] == all_results['Стандартні слоти']).all():
                    st.info("ℹ️ Оптимізація не знайшла кращого рішення. Стандартний спліт є оптимальним у межах заданих відхилень.")
                
                st.subheader("📊 Результати оптимізації")

                tab1, tab2, tab3 = st.tabs(["Загальні показники", "Деталі по СХ", "Графіки"])

                with tab1:
                    st.markdown("#### Загальна вартість за рейтинг")
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

                    st.markdown("---")
                    st.markdown("#### Деталі спліта по каналах")
                    display_df_channels = all_results[['Канал', 'СХ', 
                                                       'Стандартні слоти', 'Стандартний TRP', 'Стандартний Aff', 
                                                       'Оптимальні слоти', 'Оптимальний TRP', 'Оптимальний Aff']].copy()
                    st.dataframe(display_df_channels.set_index('Канал'))
                
                with tab2:
                    st.markdown("#### Розподіл вартості по СХ")
                    
                    sh_results_opt = all_results.groupby('СХ').agg(
                        {'Оптимальний бюджет': 'sum',
                         'Оптимальний TRP': 'sum',
                         'Оптимальний Aff': 'sum'}
                    )
                    sh_results_std = all_results.groupby('СХ').agg(
                        {'Стандартний бюджет': 'sum',
                         'Стандартний TRP': 'sum',
                         'Стандартний Aff': 'sum'}
                    )
                    
                    sh_results_opt['Ціна за Aff'] = sh_results_opt['Оптимальний бюджет'] / sh_results_opt['Оптимальний Aff']
                    sh_results_opt['Ціна за TRP'] = sh_results_opt['Оптимальний бюджет'] / sh_results_opt['Оптимальний TRP']
                    sh_results_std['Ціна за Aff'] = sh_results_std['Стандартний бюджет'] / sh_results_std['Стандартний Aff']
                    sh_results_std['Ціна за TRP'] = sh_results_std['Стандартний бюджет'] / sh_results_std['Стандартний TRP']

                    display_df_sh_costs = pd.DataFrame({
                        'СХ': sh_results_opt.index,
                        'Ціна за Aff (стандарт)': sh_results_std['Ціна за Aff'],
                        'Ціна за TRP (стандарт)': sh_results_std['Ціна за TRP'],
                        'Ціна за Aff (оптимізований)': sh_results_opt['Ціна за Aff'],
                        'Ціна за TRP (оптимізований)': sh_results_opt['Ціна за TRP']
                    })
                    st.dataframe(display_df_sh_costs.set_index('СХ').fillna(0).applymap(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x))
                
                with tab3:
                    st.markdown("#### Порівняння сплітів за часткою бюджету, загальним TRP та кількістю слотів")
                    
                    # Графік 1: Розподіл частки бюджету
                    fig_budget_share, ax_budget_share = plt.subplots(figsize=(12, 6))
                    labels = all_results['Канал']
                    std_share = (all_results['Стандартний бюджет'] / all_results['Стандартний бюджет'].sum()) * 100
                    opt_share = (all_results['Оптимальний бюджет'] / all_results['Оптимальний бюджет'].sum()) * 100
                    
                    width = 0.35
                    x = range(len(labels))
                    ax_budget_share.bar(x, std_share, width, label='Стандартний спліт', color='gray')
                    ax_budget_share.bar([p + width for p in x], opt_share, width, label='Оптимізований спліт', color='skyblue')
                    
                    ax_budget_share.set_title('Розподіл частки бюджету (%)')
                    ax_budget_share.set_ylabel('Частка (%)')
                    ax_budget_share.set_xticks([p + width / 2 for p in x])
                    ax_budget_share.set_xticklabels(labels, rotation=45, ha="right")
                    ax_budget_share.legend()
                    ax_budget_share.grid(axis='y')
                    plt.tight_layout()
                    st.pyplot(fig_budget_share)
                    
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
                    
                    # 1. Спліт (Стандартний та Оптимізований)
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

                    # 2. Вартість по СХ
                    display_df_sh_costs.set_index('СХ').to_excel(writer, sheet_name='Вартість по СХ')

                st.download_button("📥 Завантажити результати Excel", data=output.getvalue(), file_name="результати_оптимізації.xlsx")
