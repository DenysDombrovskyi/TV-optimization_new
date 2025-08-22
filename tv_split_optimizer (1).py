
import streamlit as st
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Оптимізація ТВ спліта", layout="wide")
st.title("📺 Оптимізація ТВ спліта | Dentsu X")

uploaded_file = st.file_uploader("Завантажте Excel-файл з даними", type=["xlsx"])

if uploaded_file:
    try:
        standard_df = pd.read_excel(uploaded_file, sheet_name="Сп-во", skiprows=1, engine="openpyxl")
        aff_df_raw = pd.read_excel(uploaded_file, sheet_name="Оптимізація спліта (викл)", engine="openpyxl")

        header_row_index = None
        for i in range(len(aff_df_raw)):
            row_values = aff_df_raw.iloc[i].astype(str).str.strip().str.lower().tolist()
            if 'канал' in row_values and 'aff' in row_values:
                header_row_index = i
                break

        if header_row_index is not None:
            aff_df = pd.read_excel(uploaded_file, sheet_name="Оптимізація спліта (викл)", header=header_row_index, engine="openpyxl")
            aff_df.columns = aff_df.columns.str.strip().str.lower()
            if 'канал' in aff_df.columns and 'aff' in aff_df.columns:
                aff_df = aff_df[['канал', 'aff']].copy()
                aff_df.columns = ['Канал', 'Aff']
                st.success("✅ Дані успішно завантажено!")
            else:
                st.error("❌ Не знайдено колонки 'Канал' та 'Aff' після нормалізації назв.")
                st.stop()
        else:
            st.error("❌ Не вдалося знайти рядок з колонками 'Канал' та 'Aff'. Перевірте структуру листа.")
            st.stop()

        all_data = pd.merge(standard_df, aff_df, on='Канал')
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

        missing_ba = [sh for sh in all_sh if buying_audiences.get(sh) is None]
        if missing_ba:
            st.error(f"❌ Не обрано БА для наступних СХ: {', '.join(missing_ba)}")
            st.stop()

        if st.button("🚀 Запустити оптимізацію"):
            all_data['Ціна'] = all_data.apply(lambda row: row.get(f'Ціна_{buying_audiences.get(row["СХ"])}', 0), axis=1)
            all_data['TRP'] = all_data.apply(lambda row: row.get(f'TRP_{buying_audiences.get(row["СХ"])}', 0), axis=1)
            all_results = pd.DataFrame()

            if mode == 'per_sh':
                total_standard_budget = (all_data['TRP'] * all_data['Ціна']).sum()
                for sales_house, group_df in all_data.groupby('СХ'):
                    group_standard_budget = (group_df['TRP'] * group_df['Ціна']).sum()
                    group_budget = (group_standard_budget / total_standard_budget) * total_budget
                    group_df['Стандартний бюджет'] = group_df['TRP'] * group_df['Ціна']
                    group_df['Доля по бюджету (%)'] = (group_df['Стандартний бюджет'] / group_df['Стандартний бюджет'].sum()) * 100
                    group_df['Відхилення'] = group_df['Доля по бюджету (%)'].apply(lambda x: 0.2 if x >= 10 else 0.3)
                    group_df['Нижня межа'] = group_df['Стандартний бюджет'] * (1 - group_df['Відхилення'])
                    group_df['Верхня межа'] = group_df['Стандартний бюджет'] * (1 + group_df['Відхилення'])
                    c = -group_df[goal].values
                    A_ub = [group_df['Ціна'].values]
                    b_ub = [group_budget]
                    A_lower = -pd.get_dummies(group_df['Канал']).mul(group_df['Ціна'], axis=0).values
                    b_lower = -group_df['Нижня межа'].values
                    A_upper = pd.get_dummies(group_df['Канал']).mul(group_df['Ціна'], axis=0).values
                    b_upper = group_df['Верхня межа'].values
                    A = [A_ub[0]] + list(A_lower) + list(A_upper)
                    b = b_ub + list(b_lower) + list(b_upper)
                    result = linprog(c, A_ub=A, b_ub=b, bounds=(0, None))
                    if result.success:
                        slots = result.x.round(0).astype(int)
                        group_df['Оптимальні слоти'] = slots
                        group_df['Оптимальний бюджет'] = slots * group_df['Ціна']
                        all_results = pd.concat([all_results, group_df])
            else:
                all_data['Стандартний бюджет'] = all_data['TRP'] * all_data['Ціна']
                total_standard_budget = all_data['Стандартний бюджет'].sum()
                all_data['Доля по бюджету (%)'] = (all_data['Стандартний бюджет'] / total_standard_budget) * 100
                all_data['Відхилення'] = all_data['Доля по бюджету (%)'].apply(lambda x: 0.2 if x >= 10 else 0.3)
                all_data['Нижня межа'] = all_data['Стандартний бюджет'] * (1 - all_data['Відхилення'])
                all_data['Верхня межа'] = all_data['Стандартний бюджет'] * (1 + all_data['Відхилення'])
                c = -all_data[goal].values
                A_ub = [all_data['Ціна'].values]
                b_ub = [total_budget]
                A_lower = -pd.get_dummies(all_data['Канал']).mul(all_data['Ціна'], axis=0).values
                b_lower = -all_data['Нижня межа'].values
                A_upper = pd.get_dummies(all_data['Канал']).mul(all_data['Ціна'], axis=0).values
                b_upper = all_data['Верхня межа'].values
                A = [A_ub[0]] + list(A_lower) + list(A_upper)
                b = b_ub + list(b_lower) + list(b_upper)
                result = linprog(c, A_ub=A, b_ub=b, bounds=(0, None))
                if result.success:
                    slots = result.x.round(0).astype(int)
                    all_data['Оптимальні слоти'] = slots
                    all_data['Оптимальний бюджет'] = slots * all_data['Ціна']
                    all_results = all_data

            if not all_results.empty:
                total_trp = (all_results['Оптимальні слоти'] * all_results['TRP']).sum()
                all_results['TRP_оптимізований_спліт (%)'] = (all_results['Оптимальні слоти'] * all_results['TRP'] / total_trp) * 100
                st.subheader("📊 Результати оптимізації")
                st.dataframe(all_results[['Канал', 'СХ', 'Оптимальний бюджет', 'TRP_оптимізований_спліт (%)']])

                fig, axes = plt.subplots(1, 2, figsize=(18, 8))
                labels = all_results['Канал']
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
                st.pyplot(fig)

                output = io.BytesIO()
                all_results.to_excel(output, index=False, engine='openpyxl')
                st.download_button("📥 Завантажити результати Excel", data=output.getvalue(), file_name="результати_оптимізації.xlsx")

    except Exception as e:
        st.error(f"❌ Помилка при обробці файлу: {e}")
