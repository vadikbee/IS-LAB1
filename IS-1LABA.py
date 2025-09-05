# main_app.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

# Импортируем функции из нашего файла
import lab_functions as lf

# --- Цвета и Стили в стиле Сакуры ---
BG_COLOR = "#FFF0F5"  # Нежный лавандово-розовый
BUTTON_COLOR = "#FFB6C1"  # Светло-розовый
TEXT_COLOR = "#4B0082"  # Глубокий индиго для текста
ACCENT_COLOR = "#C71585"  # Средний фиолетово-красный для акцентов
FONT_NAME = "Arial"


class SakuraApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Анализ данных в стиле Сакуры")
        self.geometry("1000x750")

        # --- Фон с изображением Сакуры ---
        try:
            bg_image = Image.open("background.jpg")
            self.bg_photo = ImageTk.PhotoImage(bg_image)
            bg_label = tk.Label(self, image=self.bg_photo)
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        except FileNotFoundError:
            self.configure(bg=BG_COLOR)  # Резервный цвет, если картинки нет

        # --- Стилизация виджетов ---
        style = ttk.Style(self)
        style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure("TNotebook.Tab", background=BUTTON_COLOR, foreground=TEXT_COLOR,
                        font=(FONT_NAME, 12), padding=[10, 5])
        style.map("TNotebook.Tab", background=[("selected", ACCENT_COLOR)], foreground=[("selected", "white")])
        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=(FONT_NAME, 11))
        style.configure("TButton", background=BUTTON_COLOR, foreground=TEXT_COLOR, font=(FONT_NAME, 11, "bold"))
        style.configure("TEntry", fieldbackground="white", foreground=TEXT_COLOR)

        # --- Создание вкладок ---
        self.notebook = ttk.Notebook(self)
        self.simple_regression_tab = ttk.Frame(self.notebook)
        self.multi_regression_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.simple_regression_tab, text="Простая Линейная Регрессия")
        self.notebook.add(self.multi_regression_tab, text="Многомерная Линейная Регрессия")
        self.notebook.pack(expand=True, fill="both", padx=20, pady=20)

        # Инициализация переменных для данных
        self.data1 = None
        self.X1 = None
        self.y1 = None

        self.data2 = None
        self.X2 = None
        self.y2 = None
        self.mu = None
        self.sigma = None

        # --- Заполнение вкладок ---
        self._create_simple_regression_widgets()
        self._create_multi_regression_widgets()

    def _create_simple_regression_widgets(self):
        tab = self.simple_regression_tab

        # --- Левая панель управления ---
        controls_frame = ttk.Frame(tab)
        controls_frame.pack(side="left", fill="y", padx=15, pady=10)

        ttk.Label(controls_frame, text="1. Загрузка данных (Service Center)", font=(FONT_NAME, 14, "bold")).pack(
            pady=10)
        ttk.Button(controls_frame, text="Выбрать файл", command=self._load_data1).pack(fill="x", pady=5)

        ttk.Separator(controls_frame, orient='horizontal').pack(fill='x', pady=20)

        ttk.Label(controls_frame, text="2. Настройки градиентного спуска", font=(FONT_NAME, 14, "bold")).pack(pady=10)

        ttk.Label(controls_frame, text="Скорость обучения (alpha):").pack()
        self.alpha1_entry = ttk.Entry(controls_frame)
        self.alpha1_entry.insert(0, "0.01")
        self.alpha1_entry.pack(fill="x", pady=2)

        ttk.Label(controls_frame, text="Количество итераций:").pack()
        self.iters1_entry = ttk.Entry(controls_frame)
        self.iters1_entry.insert(0, "2000")
        self.iters1_entry.pack(fill="x", pady=2)

        ttk.Button(controls_frame, text="Запустить анализ", command=self._run_simple_regression).pack(fill="x", pady=15)

        # --- Правая панель с графиком ---
        self.fig1, self.ax1 = plt.subplots(1, 2, figsize=(8, 4))
        plt.style.use('seaborn-v0_8-pastel')
        self.fig1.patch.set_facecolor(BG_COLOR)

        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=tab)
        self.canvas1.get_tk_widget().pack(side="right", fill="both", expand=True)

    def _load_data1(self):
        file_path = filedialog.askopenfilename(
            title="Выберите service_center.csv",
            filetypes=(("CSV Files", "*.csv"), ("All files", "*.*"))
        )
        if not file_path: return

        try:
            self.data1 = pd.read_csv(file_path, header=None, names=['population', 'profit'])
            self.X1 = self.data1.iloc[:, 0].values
            self.y1 = self.data1.iloc[:, 1].values

            # Очистка и построение начального графика
            for ax in self.ax1: ax.clear()
            self.ax1[0].scatter(self.X1, self.y1, c=ACCENT_COLOR, marker='x', label='Обучающие данные')
            self.ax1[0].set_title('Данные сервисных центров')
            self.ax1[0].set_xlabel('Население города (x10,000)')
            self.ax1[0].set_ylabel('Прибыль (x$10,000)')
            self.ax1[0].legend()
            self.canvas1.draw()

            messagebox.showinfo("Успех", "Данные успешно загружены!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {e}")

    def _run_simple_regression(self):
        if self.data1 is None:
            messagebox.showwarning("Внимание", "Сначала загрузите данные.")
            return

        alpha = float(self.alpha1_entry.get())
        iters = int(self.iters1_entry.get())

        m = len(self.y1)
        X = np.stack([np.ones(m), self.X1], axis=1)
        y = self.y1.reshape(m, 1)
        theta = np.zeros((2, 1))

        theta_final, J_history = lf.gradientDescent(X, y, theta, alpha, iters)

        # Обновляем графики
        self.ax1[0].clear()
        self.ax1[0].scatter(self.X1, self.y1, c=ACCENT_COLOR, marker='x', label='Обучающие данные')
        self.ax1[0].plot(self.X1, X.dot(theta_final), '-', label='Линейная регрессия')
        self.ax1[0].set_title('Результат регрессии')
        self.ax1[0].set_xlabel('Население города (x10,000)')
        self.ax1[0].set_ylabel('Прибыль (x$10,000)')
        self.ax1[0].legend()

        self.ax1[1].clear()
        self.ax1[1].plot(range(iters), J_history, color=TEXT_COLOR)
        self.ax1[1].set_title('Изменение функции стоимости')
        self.ax1[1].set_xlabel('Итерации')
        self.ax1[1].set_ylabel('Стоимость J')

        self.fig1.tight_layout()
        self.canvas1.draw()

    # --- Методы для второй вкладки ---
    def _create_multi_regression_widgets(self):
        tab = self.multi_regression_tab

        # --- Левая панель управления ---
        controls_frame = ttk.Frame(tab)
        controls_frame.pack(side="left", fill="y", padx=15, pady=10)

        ttk.Label(controls_frame, text="1. Загрузка данных (Apartments)", font=(FONT_NAME, 14, "bold")).pack(pady=10)
        ttk.Button(controls_frame, text="Выбрать файл", command=self._load_data2).pack(fill="x", pady=5)

        ttk.Separator(controls_frame, orient='horizontal').pack(fill='x', pady=20)

        ttk.Label(controls_frame, text="2. Запуск методов", font=(FONT_NAME, 14, "bold")).pack(pady=10)
        ttk.Button(controls_frame, text="Метод Градиентного Спуска", command=self._run_multi_gd).pack(fill="x", pady=5)
        ttk.Button(controls_frame, text="Метод Нормальных Уравнений", command=self._run_multi_normal).pack(fill="x",
                                                                                                           pady=5)

        ttk.Separator(controls_frame, orient='horizontal').pack(fill='x', pady=20)

        ttk.Label(controls_frame, text="3. Предсказание стоимости", font=(FONT_NAME, 14, "bold")).pack(pady=10)
        ttk.Label(controls_frame, text="Площадь (м²):").pack()
        self.area_entry = ttk.Entry(controls_frame)
        self.area_entry.insert(0, "60")
        self.area_entry.pack(fill="x", pady=2)

        ttk.Label(controls_frame, text="Кол-во комнат:").pack()
        self.rooms_entry = ttk.Entry(controls_frame)
        self.rooms_entry.insert(0, "3")
        self.rooms_entry.pack(fill="x", pady=2)

        self.prediction_label = ttk.Label(controls_frame, text="Цена: --", font=(FONT_NAME, 12, "bold"))
        self.prediction_label.pack(pady=15)

    def _load_data2(self):
        file_path = filedialog.askopenfilename(
            title="Выберите cost_apartments.csv",
            filetypes=(("CSV Files", "*.csv"), ("All files", "*.*"))
        )
        if not file_path: return

        try:
            self.data2 = pd.read_csv(file_path)
            self.X2 = self.data2.iloc[:, 0:2].values
            self.y2 = self.data2.iloc[:, 2].values
            messagebox.showinfo("Успех", f"Данные ({len(self.data2)} строк) успешно загружены!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {e}")

    def _run_multi_gd(self):
        if self.data2 is None:
            messagebox.showwarning("Внимание", "Сначала загрузите данные.")
            return

        X_norm, self.mu, self.sigma = lf.featureNormalize(self.X2)
        m = len(self.y2)
        X = np.hstack([np.ones((m, 1)), X_norm])
        y = self.y2.reshape(m, 1)
        theta = np.zeros((X.shape[1], 1))

        theta_final, _ = lf.gradientDescent(X, y, theta, 0.01, 1000)

        # Предсказываем
        area = float(self.area_entry.get())
        rooms = float(self.rooms_entry.get())

        price_vec = np.array([area, rooms])
        price_vec_norm = (price_vec - self.mu) / self.sigma
        price_vec_final = np.hstack([np.ones(1), price_vec_norm])

        prediction = price_vec_final.dot(theta_final)[0]
        self.prediction_label.config(text=f"Цена (ГС): {prediction:,.0f} руб.".replace(",", " "))

    def _run_multi_normal(self):
        if self.data2 is None:
            messagebox.showwarning("Внимание", "Сначала загрузите данные.")
            return

        m = len(self.y2)
        X = np.hstack([np.ones((m, 1)), self.X2])
        y = self.y2.reshape(m, 1)

        theta_final = lf.normalEqn(X, y)

        # Предсказываем
        area = float(self.area_entry.get())
        rooms = float(self.rooms_entry.get())

        price_vec = np.array([1, area, rooms])
        prediction = price_vec.dot(theta_final)[0]
        self.prediction_label.config(text=f"Цена (НУ): {prediction:,.0f} руб.".replace(",", " "))


if __name__ == "__main__":
    app = SakuraApp()
    app.mainloop()