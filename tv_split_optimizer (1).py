import streamlit as st
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import io

# --- Функції для валідації та оптимізації ---

def validate_excel_file(df_standard, df_aff):
    """
    Перевіряє, чи існують необхідні стовпці в завантажених датафреймах.
    Повертає True, якщо валідація успішна, інакше - False.
    """
    required_cols_standard = ['Канал', 'СХ']
    required_cols_aff = ['Канал', 'Aff']

    for col in required_cols_standard:
        if col not in df_standard.columns:
            st.error(f"❌ Помилка: В аркуші 'Сп-во' відсутній обов'язковий стовпчик '{col}'.")
            return False
            
    for col in required_cols_aff:
        if col not in df_aff.columns:
            st.error(f"❌ Помилка: В аркуші 'Оптимізація спліта (викл)' відсутній обов'язковий стовпчик '{col}'.")
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
    df['Стандартний Aff'] = df['Aff']
    df['Стандартний TRP'] = df['TRP']
    
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
                
                # Використовуємо задані користувачем відхилення
                group_df['Нижня межа TRP'] = group_df['Стандартна доля TRP'] * (1 - group_df['Мінімальне відхилення'] / 100)
                group_df['Верхня межа TRP'] = group_df['Стандартна доля TRP'] * (1 + group_df['Максимальне відхилення'] / 100)
                
                # Завдання для linprog
                c = -group_df[goal].values
                A_upper = pd.get_dummies(group_df['Канал']).mul(group_df['TRP'], axis=0).values
                b_upper = group_df['Верхня межа TRP'].values * group_standard_trp
                b_lower = group_df['Нижня межа TRP'].values * group_standard_trp
                
                A_lower = -A_upper
                b_lower = -b_lower
                
                A = list(A_upper) + list(A_lower)
                b = list(b_upper) + list(b_lower)
                
                # Кожен канал повинен мати хоча б 1 слот
                bounds = [(1, None) for _ in range(len(group_df))]

                result = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
                
                if result.success:
                    slots = result.x.round(0).astype(int)
                    group_df['Оптимальні слоти'] = slots
                    group_df['Оптимальний TRP'] = slots * group_df['TRP']
                    group_df['Оптимальний Aff'] = slots * group_df['Aff']
                    group_df['Оптимальний TRP (%)'] = (group_df['Оптимальний TRP'] / group_df['Оптимальний TRP'].sum()) * 100
                    all_results = pd.concat([all_results, group_df])
                else:
                    st.warning(f"Оптимізація для СХ {sales_house} не вдалася: {result.message}")
                    
    else:  # mode == 'total'
        with st.spinner('Проводимо загальну оптимізацію...'):
            total_standard_trp = df['Стандартний TRP'].sum()
            df['Стандартна доля TRP'] = df['Стандартний TRP'] / total_standard_trp
            
            # Використовуємо задані користувачем відхилення
            df['Нижня межа TRP'] = df['Стандартна доля TRP'] * (1 - df['Мінімальне відхилення'] / 100)
            df['Верхня межа TRP'] = df['Стандартна доля TRP'] * (1 + df['Максимальне відхилення'] / 100)

            # Завдання для linprog
            c = -df[goal].values
            A_upper = pd.get_dummies(df['Канал']).mul(df['TRP'], axis=0).values
            b_upper = df['Верхня межа TRP'].values * total_standard_trp
            b_lower = df['Нижня межа TRP'].values * total_standard_trp
            
            A_lower = -A_upper
            b_lower = -b_lower

            A = list(A_upper) + list(A_lower)
            b = list(b_upper) + list(b_lower)

            bounds = [(1, None) for _ in range(len(df))]
            result = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
            
            if result.success:
                slots = result.x.round(0).astype(int)
                df['Оптимальні слоти'] = slots
                df['Оптимальний TRP'] = slots * df['TRP']
                df['Оптимальний Aff'] = slots * df['Aff']
                df['Оптимальний TRP (%)'] = (df['Оптимальний TRP'] / df['Оптимальний TRP'].sum()) * 100
                all_results = df
            else:
                st.error(f"❌ Загальна оптимізація не вдалася: {result.message}")
    
    return all_results

# --- Основна частина програми Streamlit ---

st.set_page_config(page_title="Оптимізація ТВ спліта", layout="wide")
st.title("📺 Оптимізація ТВ спліта | Dentsu X")

uploaded_file = st.file_uploader("Завантажте Excel-файл з даними", type=["xlsx"])

# Умовний бюджет для порівняльних розрахунків
total_budget_assumption = 1_000_000

if uploaded_file:
    try:
        with st.spinner('Завантаження та обробка даних...'):
            standard_df = pd.read_excel(uploaded_file, sheet_name="Сп-во", skiprows=2, engine="openpyxl")
            aff_df = pd.read_excel(uploaded_file, sheet_name="Оптимізація спліта (викл)", skiprows=7, engine="openpyxl")
            aff_df = aff_df.iloc[:, [0, 1]].copy()
            aff_df.columns = ['Канал', 'Aff']
            
            if not validate_excel_file(standard_df, aff_df):
                st.stop()

            all_data = pd.merge(standard_df, aff_df, on='Канал', how='inner')
            
            st.success("✅ Дані успішно завантажено!")
            
    except FileNotFoundError:
        st.error("❌ Помилка: Не знайдено необхідні аркуші в файлі. Переконайтеся, що файл містить аркуші 'Сп-во' та 'Оптимізація спліта (викл)'.")
    except KeyError as e:
        st.error(f"❌ Помилка: У файлі відсутній необхідний стовпець: {e}.")
    except Exception as e:
        st.error(f"❌ Сталася неочікувана помилка: {e}")
    else:
        all_sh = all_data['СХ'].unique()
        all_ba = [col.replace('Ціна_', '') for col in all_data.columns if 'Ціна_' in col]

        st.header("🔧 Налаштування оптимізації")
        st.info(f"Для порівняння фінансових показників програма використовує умовний бюджет **{total_budget_assumption:,} грн**.")
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
                all_results['Стандартний бюджет'] = all_results['Стандартний TRP'] * all_results['Ціна']
                
                # Розрахунок показників для порівняння, використовуючи умовний бюджет
                total_budget_opt_raw = all_results['Оптимальний бюджет'].sum()
                total_budget_std = all_results['Стандартний бюджет'].sum()
                
                # Масштабування, щоб загальний бюджет дорівнював умовному
                if total_budget_opt_raw > 0:
                    scale_factor = total_budget_assumption / total_budget_opt_raw
                    total_trp_opt_scaled = all_results['Оптимальний TRP'].sum() * scale_factor
                    total_aff_opt_scaled = all_results['Оптимальний Aff'].sum() * scale_factor
                else:
                    st.error("Не вдалося розрахувати масштабовані дані. Загальний оптимальний бюджет дорівнює 0.")
                    st.stop()
                
                # Розрахунок вартості за рейтинг для загального спліта
                total_trp_std = all_results['Стандартний TRP'].sum()
                total_aff_std = all_results['Стандартний Aff'].sum()
                
                cpt_opt = total_budget_assumption / total_aff_opt_scaled if total_aff_opt_scaled > 0 else 0
                cpp_opt = total_budget_assumption / total_trp_opt_scaled if total_trp_opt_scaled > 0 else 0
                cpt_std = total_budget_std / total_aff_std if total_aff_std > 0 else 0
                cpp_std = total_budget_std / total_trp_std if total_trp_std > 0 else 0

                st.subheader("📊 Результати оптимізації")
                
                st.markdown("#### Загальна вартість за рейтинг (на основі умовного бюджету)")
                col1, col2 = st.columns(2)
                with col1:
                    st.info("**Стандартний спліт**")
                    st.metric("Ціна за Aff", f"{cpt_std:,.2f} грн")
                    st.metric("Ціна за TRP", f"{cpp_std:,.2f} грн")
                with col2:
                    st.success("**Оптимізований спліт**")
                    st.metric("Ціна за Aff", f"{cpt_opt:,.2f} грн")
                    st.metric("Ціна за TRP", f"{cpp_opt:,.2f} грн")

                st.markdown("---")
                st.subheader("Деталі оптимізованого спліта (масштабовано)")
                st.metric("Загальний TRP (масштабовано)", f"{total_trp_opt_scaled:,.2f}")
                st.metric("Загальний Aff (масштабовано)", f"{total_aff_opt_scaled:,.2f}")
                
                # Створення масштабованих колонок для відображення
                all_results['Оптимальні слоти (масштаб)'] = (all_results['Оптимальні слоти'] * scale_factor).round(0).astype(int)
                all_results['Оптимальний бюджет (масштаб)'] = all_results['Оптимальні слоти (масштаб)'] * all_results['Ціна']
                all_results['Оптимальний TRP (масштаб)'] = all_results['Оптимальні слоти (масштаб)'] * all_results['TRP']
                all_results['Оптимальний Aff (масштаб)'] = all_results['Оптимальні слоти (масштаб)'] * all_results['Aff']
                all_results['Оптимальний TRP (%)'] = (all_results['Оптимальний TRP (масштаб)'] / all_results['Оптимальний TRP (масштаб)'].sum()) * 100
                
                st.dataframe(all_results[['Канал', 'СХ', 'Оптимальні слоти (масштаб)', 'Оптимальний TRP (масштаб)', 'Оптимальний Aff (масштаб)', 'Оптимальний TRP (%)']])

                # Візуалізація результатів
                fig, axes = plt.subplots(1, 2, figsize=(18, 8))
                labels = all_results['Канал']
                
                if 'Стандартний TRP' in all_results.columns:
                    std_share = (all_results['Стандартний TRP'] / all_results['Стандартний TRP'].sum()) * 100
                    opt_share = (all_results['Оптимальний TRP (масштаб)'] / all_results['Оптимальний TRP (масштаб)'].sum()) * 100
                    x = range(len(labels))
                    width = 0.35
                    axes[0].bar(x, std_share, width, label='Стандартний спліт', color='gray')
                    axes[0].bar([p + width for p in x], opt_share, width, label='Оптимізований спліт', color='skyblue')
                    axes[0].set_title('Частка TRP (%)')
                    axes[0].set_xticks([p + width / 2 for p in x])
                    axes[0].set_xticklabels(labels, rotation=45, ha="right")
                    axes[0].legend()
                    axes[0].grid(axis='y')

                axes[1].bar(labels, all_results['Оптимальні слоти (масштаб)'], color='skyblue')
                axes[1].set_title('Кількість слотів')
                axes[1].set_xticklabels(labels, rotation=45, ha="right")
                axes[1].grid(axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)

                # Кнопка для завантаження результатів
                output = io.BytesIO()
                all_results.to_excel(output, index=False, engine='openpyxl')
                st.download_button("📥 Завантажити результати Excel", data=output.getvalue(), file_name="результати_оптимізації.xlsx")
