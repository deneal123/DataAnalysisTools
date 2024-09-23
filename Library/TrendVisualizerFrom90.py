import matplotlib.pyplot as plt
import numpy as np


class TrendVisualizer:
    """
    Класс для визуализации музыкальных параметров с добавлением линии тренда.
    """

    def __init__(self, metrics):
        """
        Инициализация с данными о метриках.

        Args:
        metrics (pd.DataFrame): данные с рассчитанными средними значениями параметров по годам (с 1990).
        """
        self.metrics = metrics

    @staticmethod
    def _plot_with_trend(x, y, label, color, ylabel):
        """
        Внутренняя функция для построения графика с линией тренда.

        Args:
        x (pd.Series): ось X (года).
        y (pd.Series): ось Y (значения метрики).
        label (str): название метрики.
        color (str): цвет линии.
        ylabel (str): подпись оси Y.
        """
        plt.plot(x, y, label=label, color=color)

        # Линейная регрессия (тренд)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "--", color='gray', label=f'{label} Trend')

        plt.xlabel("Год выпуска")
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

    def visualize_tempo(self):
        """Визуализирует изменение темпа по годам с трендлайном."""
        plt.figure(figsize=(12, 8))
        self._plot_with_trend(self.metrics['year'], self.metrics['tempo'], 'Tempo', 'b', 'Средний темп (BPM)')

    def visualize_duration(self):
        """Визуализирует изменение длительности треков по годам с трендлайном."""
        plt.figure(figsize=(12, 8))
        self._plot_with_trend(self.metrics['year'], self.metrics['duration_ms'], 'Duration', 'g',
                              'Средняя длительность (мс)')

    def visualize_instrumentalness(self):
        """Визуализирует изменение инструментальности по годам с трендлайном."""
        plt.figure(figsize=(12, 8))
        self._plot_with_trend(self.metrics['year'], self.metrics['instrumentalness'], 'Instrumentalness', 'r',
                              'Средняя инструментальность')

    def visualize_danceability(self):
        """Визуализирует изменение танцевальности по годам с трендлайном."""
        plt.figure(figsize=(12, 8))
        self._plot_with_trend(self.metrics['year'], self.metrics['danceability'], 'DanceAbility', 'r',
                              'Средняя танцевальность')

    def visualize_energy(self):
        """Визуализирует изменение энергичности по годам с трендлайном."""
        plt.figure(figsize=(12, 8))
        self._plot_with_trend(self.metrics['year'], self.metrics['energy'], 'Energy', 'r',
                              'Средняя энергичность')

    def visualize_acousticness(self):
        """Визуализирует изменение акустичности по годам с трендлайном."""
        plt.figure(figsize=(12, 8))
        self._plot_with_trend(self.metrics['year'], self.metrics['acousticness'], 'Acousticness', 'r',
                              'Средняя акустичность')

    def visualize_liveness(self):
        """Визуализирует изменение вероятности живых выступлений по годам с трендлайном."""
        plt.figure(figsize=(12, 8))
        self._plot_with_trend(self.metrics['year'], self.metrics['liveness'], 'Liveness', 'r',
                              'Средняя вероятность живых выступлений')




