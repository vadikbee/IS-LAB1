# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --- Функции из лабораторной работы (без изменений) ---

def computeCost(X, y, theta):
    m = len(y)
    if y.ndim == 1: y = y.reshape(-1, 1)
    predictions = X.dot(theta)
    sqrErrors = (predictions - y) ** 2
    J = (1 / (2 * m)) * np.sum(sqrErrors)
    return J


def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    if y.ndim == 1: y = y.reshape(-1, 1)
    for i in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        delta = (1 / m) * (X.T.dot(errors))
        theta = theta - (alpha * delta)
        J_history[i] = computeCost(X, y, theta)
    return theta, J_history


def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def normalEqn(X, y):
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta


# --- НОВЫЙ ДИЗАЙН "НОЧНАЯ САКУРА" ---

st.set_page_config(page_title="Лабораторная по регрессии", layout="wide")

# Глобальный блок стилей для темной темы
st.markdown("""
<style>
    /* Основной фон с темным градиентом */
    .stApp {
        background: linear-gradient(135deg, #1D2B64 0%, #0E1632 100%);
        color: #F0F2F6; /* Светло-серый цвет для основного текста */
    }

    /* Стилизация "стеклянных" контейнеров-карточек */
    .block-container {
        background-color: rgba(44, 62, 80, 0.7); /* Полупрозрачный темно-синий */
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    /* Стили для заголовков */
    h1, h2, h3 {
        color: #FFFFFF; /* Ярко-белый цвет */
    }

    /* Стилизация вкладок */
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #BDC3C7; /* Неактивный цвет текста вкладки */
    }
    .stTabs [aria-selected="true"] {
        background-color: #E91E63; /* Яркий розовый для активной вкладки */
        color: white;
        font-weight: bold;
    }

    /* Стилизация виджета загрузки файлов */
    .stFileUploader label {
        color: #ECF0F1 !important;
    }
</style>
""", unsafe_allow_html=True)

# Задаем темный стиль для графиков Matplotlib
plt.style.use('dark_background')

# --- Интерфейс приложения ---

st.markdown("<h1 style='text-align: center;'>🌸 Анализ методом линейной регрессии (Night Theme) 🌸</h1>",
            unsafe_allow_html=True)
st.markdown("---")

tab1, tab2 = st.tabs(["📈 Простая линейная регрессия", "🏘️ Многомерная линейная регрессия"])

# --- ВКЛАДКА 1 ---
with tab1:
    st.header("Прогнозирование прибыли сервисного центра")

    with st.container():
        st.markdown('<div class="block-container">', unsafe_allow_html=True)
        uploaded_file_1 = st.file_uploader("Загрузите файл service_center.csv", type="csv", key="uploader1")
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file_1:
        data1 = pd.read_csv(uploaded_file_1)
        if len(data1.columns) == 2 and 'population' not in data1.columns:
            data1.columns = ['population', 'profit']

        X1_orig, y1 = data1['population'].values, data1['profit'].values

        with st.container():
            st.markdown('<div class="block-container">', unsafe_allow_html=True)
            st.subheader("Визуализация исходных данных")
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            fig1.patch.set_alpha(0.0)
            ax1.set_facecolor('none')
            ax1.scatter(X1_orig, y1, c='#E91E63', marker='o', s=50, label='Обучающие данные')
            ax1.set_xlabel('Население города (x10,000)', fontsize=12)
            ax1.set_ylabel('Прибыль (x$10,000)', fontsize=12)
            ax1.set_title('Зависимость прибыли от населения', fontsize=16)
            ax1.legend(fontsize=12)
            st.pyplot(fig1)
            st.markdown('</div>', unsafe_allow_html=True)

    st.sidebar.header("Настройки простой регрессии")
    alpha1 = st.sidebar.slider("Скорость обучения (α)", 0.001, 0.03, 0.01, step=0.001, key="alpha1")
    iters1 = st.sidebar.slider("Количество итераций", 100, 3000, 1500, step=100, key="iters1")

    if uploaded_file_1 and st.sidebar.button("🚀 Рассчитать регрессию"):
        with st.container():
            st.markdown('<div class="block-container">', unsafe_allow_html=True)
            st.subheader("Результаты анализа")
            with st.spinner('Выполняется градиентный спуск...'):
                m = len(y1)
                X1 = np.stack([np.ones(m), X1_orig], axis=1)
                theta_final, J_history = gradientDescent(X1, y1, np.zeros((2, 1)), alpha1, iters1)

            st.success(f"Оптимальные параметры: θ₀ = {theta_final[0][0]:.4f}, θ₁ = {theta_final[1][0]:.4f}")

            fig_res, (ax_reg, ax_cost) = plt.subplots(1, 2, figsize=(14, 6))
            fig_res.patch.set_alpha(0.0)

            ax_reg.scatter(X1_orig, y1, c='#E91E63', marker='o', alpha=0.7, label='Данные')
            ax_reg.plot(X1_orig, X1.dot(theta_final), '-', label='Линейная регрессия', color='#00FFFF', linewidth=2)
            ax_reg.set_title('Результат регрессии', fontsize=16)
            ax_reg.legend(fontsize=12)

            ax_cost.plot(range(len(J_history)), J_history, color='#00FFFF')
            ax_cost.set_title('Сходимость функции стоимости', fontsize=16)

            st.pyplot(fig_res)
            st.markdown('</div>', unsafe_allow_html=True)

# --- ВКЛАДКА 2 ---
with tab2:
    st.header("Оценка стоимости квартиры")

    with st.container():
        st.markdown('<div class="block-container">', unsafe_allow_html=True)
        uploaded_file_2 = st.file_uploader("Загрузите файл cost_apartments.csv", type="csv", key="uploader2")
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file_2:
        with st.container():
            st.markdown('<div class="block-container">', unsafe_allow_html=True)
            data2 = pd.read_csv(uploaded_file_2)
            if len(data2.columns) == 3 and 'squera' not in data2.columns:
                data2.columns = ['squera', 'number_rooms', 'price']

            st.sidebar.header("Параметры для предсказания")
            area = st.sidebar.number_input("Площадь (м²):", min_value=10, max_value=200, value=60)
            rooms = st.sidebar.number_input("Кол-во комнат:", min_value=1, max_value=10, value=3)

            st.subheader("Сравнение методов расчета")
            col1, col2 = st.columns(2)

            X2_orig, y2 = data2[['squera', 'number_rooms']].values, data2['price'].values

            with col1:
                st.markdown("#### Метод градиентного спуска")
                with st.spinner('Расчет...'):
                    X_norm, mu, sigma = featureNormalize(X2_orig)
                    X2_gd = np.hstack([np.ones((len(y2), 1)), X_norm])
                    theta_gd, _ = gradientDescent(X2_gd, y2, np.zeros((3, 1)), 0.1, 500)
                    price_vec_norm = (np.array([area, rooms]) - mu) / sigma
                    price_vec_final = np.hstack([1, price_vec_norm])
                    prediction_gd = price_vec_final.dot(theta_gd)[0]
                st.metric(label="Предсказанная цена", value=f"{prediction_gd:,.0f} руб.".replace(",", " "))

            with col2:
                st.markdown("#### Метод нормальных уравнений")
                with st.spinner('Расчет...'):
                    X2_ne = np.hstack([np.ones((len(y2), 1)), X2_orig])
                    theta_ne = normalEqn(X2_ne, y2)
                    price_vec_ne = np.array([1, area, rooms])
                    prediction_ne = price_vec_ne.dot(theta_ne)
                st.metric(label="Предсказанная цена", value=f"{prediction_ne:,.0f} руб.".replace(",", " "))

            st.markdown('</div>', unsafe_allow_html=True)