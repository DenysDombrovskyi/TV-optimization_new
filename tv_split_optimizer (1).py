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
            st.error(f"❌ Помилка: В аркуші 'Сп-во' відсутній обов'язковий стовпець '{col}'.")
            return False
            
    for col in required_cols_aff:
        if col not in df_aff.columns:
            st.error(f"❌ Помилка: В аркуші 'Оптимізація спліта (викл)' відсутній обов'язковий стовпець '{col}'.")
            return False
    
    return True

def run_optimization(df, total_budget, goal, mode, buying_audiences):
    """
    Виконує оптимізацію ТВ-спліта на основі даних і налаштувань.
    Повертає датафрейм з результатами оптимізації.
    """
    # Додаємо стовпці з цінами та TRP для обраної БА
    df['Ціна'] = df.apply(lambda row: row.get(f'Ціна_{buying_audiences.get(row["СХ"], "")}', 0), axis=1)
    df['TRP'] = df.apply(lambda row: row.get(f'TRP_{buying_audiences.get(row["СХ"], "")}', 0), axis=1)
    
    # Видаляємо рядок, який фільтрував канали з нульовими значеннями
    # Тепер програма буде обробляти всі канали.

    all_results = pd.DataFrame()

    if mode == 'per_sh':
        with st.spinner('Проводимо оптимізацію за кожним СХ...'):
            total_standard_budget = (df['TRP'] * df['Ціна']).sum()
            for sales_house, group_df in df.groupby('СХ'):
                group_standard_budget = (group_df['TRP'] * group_df['Ціна']).sum()
                if total_standard_budget == 0:
                    st.warning(f"Недостатньо даних для СХ {sales_house}. Пропускаємо...")
                    continue
                group_budget = (group_standard_budget / total_standard_budget) * total_budget
                
                group_df['Стандартний бюджет'] = group_df['TRP'] * group_df['Ціна']
                group_df['Доля по бюджету (%)'] = (group_df['Стандартний бюджет'] / group_df['Стандартний бюджет'].sum()) * 100
                
                # Логіка відхилень на основі частки бюджету
                group_df['Відхилення'] = group_df['Доля по бюджету (%)'].apply(lambda x: 0.2 if x >= 10 else 0.3)
                
                group_df['Нижня межа'] = group_df['Стандартний бюджет'] * (1 - group_df['Відхилення'])
                group_df['Верхня межа'] = group_df['Стандартний бюджет'] * (1 + group_df['Відхилення'])
                
                # Завдання для linprog
                c = -group_df[goal].values
                A_ub = [group_df['Ціна'].values]
                b_ub = [group_budget]
                A_lower = -pd.get_dummies(group_df['Канал']).mul(group_df['Ціна'], axis=0).values
                b_lower = -group_df['Нижня межа'].values
                A_upper = pd.get_dummies(group_df['Канал']).mul(group_df['Ціна'], axis=0).values
                b_upper = group_df['Верхня межа'].values
                A = A_ub + list(A_lower) + list(A_upper)
                b = b_ub + list(b_lower) + list(b_upper)
                
                result = linprog(c, A_ub=A, b_ub=b, bounds=(0, None))
                
                if result.success:
                    slots = result.x.round(0).astype(int)
                    group_df['Оптимальні слоти'] = slots
                    group_df['Оптимальний бюджет'] = slots * group_df['Ціна']
                    all_results = pd.concat([all_results, group_df])
                else:
                    st.warning(f"Оптимізація для СХ {sales_house} не вдалася: {result.message}")
                    
    else:  # mode == 'total'
        with st.spinner('Проводимо загальну оптимізацію...'):
            df['Стандартний бюджет'] = df['TRP'] * df['Ціна']
            total_standard_budget = df['Стандартний бюджет'].sum()
            df['Доля по бюджету (%)'] = (df['Стандартний бюджет'] / total_standard_budget) * 100
            
            # Логіка відхилень на основі частки бюджету
            df['Відхилення'] = df['Доля по бюджету (%)'].apply(lambda x: 0.2 if x >= 10 else 0.3)
            
            df['Нижня межа'] = df['Стандартний бюджет'] * (1 - df['Відхилення'])
            df['Верхня межа'] = df['Стандартний бюджет'] * (1 + df['Відхилення'])
            
            # Завдання для linprog
            c = -df[goal].values
            A_ub = [df['Ціна'].values]
            b_ub = [total_budget]
            A_lower = -pd.get_dummies(df['Канал']).mul(df['Ціна'], axis=0).values
            b_lower = -df['Нижня межа'].values
            A_upper = pd.get_dummies(df['Канал']).mul(df['Ціна'], axis=0).values
            b_upper = df['Верхня межа'].values
            A = A_ub + list(A_lower) + list(A_upper)
            b = b_ub + list(b_lower) + list(b_upper)

            result = linprog(c, A_ub=A, b_ub=b, bounds=(0, None))
            
            if result.success:
                slots = result.x.round(0).astype(int)
                df['Оптимальні слоти'] = slots
                df['Оптимальний бюджет'] = slots * df['Ціна']
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
            standard_df = pd.read_excel(uploaded_file, sheet_name="Сп-во", skiprows=1, engine="openpyxl")
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
        total_budget = st.number_input("Загальний бюджет (грн)", min_value=1000, value=500000, step=1000)
        goal = st.selectbox("Мета оптимізації", ['Aff', 'TRP'])
        mode = st.selectbox("Режим оптимізації", ['total', 'per_sh'])
        
        st.subheader("🎯 Вибір БА для кожного СХ")
        buying_audiences = {}
        for sh in all_sh:
            ba = st.selectbox(f"СХ: {sh}", all_ba, key=sh)
            buying_audiences[sh] = ba

        if st.button("🚀 Запустити оптимізацію"):
            all_results = run_optimization(all_data.copy(), total_budget, goal, mode, buying_audiences)
            
            if not all_results.empty:
                # Розрахунок підсумкових даних
                total_trp = (all_results['Оптимальні слоти'] * all_results['TRP']).sum()
                total_budget_opt = all_results['Оптимальний бюджет'].sum()
                
                st.subheader("📊 Результати оптимізації")
                st.metric("Загальний TRP (оптимізований)", f"{total_trp:,.2f}")
                st.metric("Витрачений бюджет", f"{total_budget_opt:,.2f} грн")

                all_results['TRP_оптимізований_спліт (%)'] = (all_results['Оптимальні слоти'] * all_results['TRP'] / total_trp) * 100
                st.dataframe(all_results[['Канал', 'СХ', 'Оптимальний бюджет', 'TRP_оптимізований_спліт (%)']])

                # Візуалізація результатів
                fig, axes = plt.subplots(1, 2, figsize=(18, 8))
                labels = all_results['Канал']
                
                if 'Стандартний бюджет' in all_results.columns:
                    std_share = (all_results['Стандартний бюджет'] / all_results['Стандартний бюджет'].sum()) * 100
                    opt_share = (all_results['Оптимальний бюджет'] / all_results['Оптимальний бюджет'].sum()) * 100
                    x = range(len(labels))
                    width = 0.35
                    axes[0].bar(x, std_share, width, label='Стандартний спліт', color='gray')
                    axes[0].bar([p + width for p in x], opt_share, width, label='Оптимізований спліт', color='skyblue')
                    axes[0].set_title('Частка бюджету (%)')
                    axes[0].set_xticks([p + width / 2 for p in x])
                    axes[0].set_xticklabels(labels, rotation=45, ha="right")
                    axes[0].legend()
                    axes[0].grid(axis='y')

                axes[1].bar(labels, all_results['Оптимальні слоти'], color='skyblue')
                axes[1].set_title('Кількість слотів')
                axes[1].set_xticklabels(labels, rotation=45, ha="right")
                axes[1].grid(axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)

                # Кнопка для завантаження результатів
                output = io.BytesIO()
                all_results.to_excel(output, index=False, engine='openpyxl')
                st.download_button("📥 Завантажити результати Excel", data=output.getvalue(), file_name="результати_оптимізації.xlsx")
