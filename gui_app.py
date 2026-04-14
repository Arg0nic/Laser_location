from __future__ import annotations

import json
import queue
import threading
import traceback
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from simulation import SimulationConfig, SimulationResults, export_results, run_simulation
from simulation.utils import create_run_output_dir, format_optional_distance


DEFAULT_CONFIG_PATH = Path("configs/default_config.json")


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    label: str
    parser: Callable[[str], Any] | type
    section: str
    widget: str = "entry"


PARAMETER_SPECS = [
    ParameterSpec("L_min", "L_min, м", float, "Сетка расстояний"),
    ParameterSpec("L_max", "L_max, м", float, "Сетка расстояний"),
    ParameterSpec("dL", "dL, м", float, "Сетка расстояний"),
    ParameterSpec("N", "N испытаний", int, "Монте-Карло"),
    ParameterSpec("M", "M импульсов", int, "Монте-Карло"),
    ParameterSpec("random_seed", "random_seed", int, "Монте-Карло"),
    ParameterSpec("theta_0", "theta_0, рад", float, "Геометрия"),
    ParameterSpec("d_target", "d_target, м", float, "Геометрия"),
    ParameterSpec("eta_min", "eta_min", float, "Сигнал"),
    ParameterSpec("A0", "A0", float, "Сигнал"),
    ParameterSpec("b", "b", float, "Сигнал"),
    ParameterSpec("sigma_A", "sigma_A", float, "Сигнал"),
    ParameterSpec("T", "Порог T", float, "Сигнал"),
    ParameterSpec("alpha", "alpha", float, "Сигнал"),
    ParameterSpec("sigma_w_value", "sigma_w_value, м", float, "Блуждание луча"),
    ParameterSpec("sigma_w_slope", "sigma_w_slope", float, "Блуждание луча"),
    ParameterSpec("p_required", "p_required", float, "Критерий"),
]


class LaserSimulationGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Симулятор лазерного дальномера")
        self.root.geometry("1450x860")
        self.root.minsize(1180, 720)

        self.current_config = self._load_initial_config()
        self.current_results: SimulationResults | None = None

        self.result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None

        self.variables: dict[str, tk.Variable] = {}
        self.widgets: dict[str, ttk.Widget] = {}
        self.status_var = tk.StringVar(value="Готово к запуску")
        self.summary_var = tk.StringVar(value="Задайте параметры и нажмите «Запустить моделирование».")

        self.run_button: ttk.Button | None = None
        self.save_button: ttk.Button | None = None

        self.figure = Figure(figsize=(8.8, 6.6), dpi=100)
        self.ax_probability = self.figure.add_subplot(211)
        self.ax_spot = self.figure.add_subplot(212)
        self.canvas: FigureCanvasTkAgg | None = None

        self._build_layout()
        self._apply_config_to_form(self.current_config)
        self._draw_placeholder()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ttk.Frame(self.root, padding=12)
        controls.grid(row=0, column=0, sticky="nsew")
        controls.rowconfigure(1, weight=1)

        ttk.Label(
            controls,
            text="Параметры эксперимента",
            font=("Segoe UI", 15, "bold"),
        ).grid(row=0, column=0, sticky="w")

        self._build_scrollable_form(controls)
        self._build_action_bar(controls)

        viewer = ttk.Frame(self.root, padding=(0, 12, 12, 12))
        viewer.grid(row=0, column=1, sticky="nsew")
        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(1, weight=1)

        summary_frame = ttk.LabelFrame(viewer, text="Сводка")
        summary_frame.grid(row=0, column=0, sticky="ew", padx=(0, 0), pady=(0, 10))
        summary_frame.columnconfigure(0, weight=1)
        ttk.Label(
            summary_frame,
            textvariable=self.summary_var,
            justify="left",
            anchor="w",
            padding=10,
        ).grid(row=0, column=0, sticky="ew")

        plot_frame = ttk.LabelFrame(viewer, text="Графики")
        plot_frame.grid(row=1, column=0, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")

        status_bar = ttk.Label(
            self.root,1
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
            padding=(8, 4),
        )
        status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _build_scrollable_form(self, parent: ttk.Frame) -> None:
        container = ttk.Frame(parent)
        container.grid(row=1, column=0, sticky="nsew", pady=(10, 10))
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        canvas = tk.Canvas(container, highlightthickness=0, width=390)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas)

        window_id = canvas.create_window((0, 0), window=content, anchor="nw")

        def on_configure(_: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        content.bind("<Configure>", on_configure)
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        section_frames: dict[str, ttk.LabelFrame] = {}
        for spec in PARAMETER_SPECS:
            if spec.section not in section_frames:
                frame = ttk.LabelFrame(content, text=spec.section, padding=10)
                frame.pack(fill="x", expand=True, pady=(0, 10))
                frame.columnconfigure(1, weight=1)
                section_frames[spec.section] = frame

            frame = section_frames[spec.section]
            row = len(frame.grid_slaves()) // 2
            ttk.Label(frame, text=spec.label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)

            if spec.widget == "check":
                variable = tk.BooleanVar(value=False)
                widget = ttk.Checkbutton(frame, variable=variable)
                widget.grid(row=row, column=1, sticky="w", pady=4)
            else:
                variable = tk.StringVar()
                widget = ttk.Entry(frame, textvariable=variable)
                widget.grid(row=row, column=1, sticky="ew", pady=4)

            self.variables[spec.name] = variable
            self.widgets[spec.name] = widget

    def _build_action_bar(self, parent: ttk.Frame) -> None:
        actions = ttk.Frame(parent)
        actions.grid(row=2, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        actions.columnconfigure(2, weight=1)
        actions.columnconfigure(3, weight=1)

        ttk.Button(actions, text="Загрузить JSON", command=self._load_config_file).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 6),
        )
        ttk.Button(actions, text="Сбросить значения", command=self._reset_defaults).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=6,
        )
        self.save_button = ttk.Button(
            actions,
            text="Сохранить результаты",
            command=self._save_results,
            state="disabled",
        )
        self.save_button.grid(row=0, column=2, sticky="ew", padx=6)
        self.run_button = ttk.Button(actions, text="Запустить моделирование", command=self._start_run)
        self.run_button.grid(row=0, column=3, sticky="ew", padx=(6, 0))

    def _load_initial_config(self) -> SimulationConfig:
        if DEFAULT_CONFIG_PATH.exists():
            try:
                return SimulationConfig.from_json(DEFAULT_CONFIG_PATH)
            except Exception:
                pass
        return SimulationConfig()

    def _apply_config_to_form(self, config: SimulationConfig) -> None:
        payload = config.to_dict()
        for spec in PARAMETER_SPECS:
            value = payload[spec.name]
            variable = self.variables[spec.name]
            if isinstance(variable, tk.BooleanVar):
                variable.set(bool(value))
            elif spec.name == "random_seed" and value is None:
                variable.set("")
            else:
                variable.set(str(value))

    def _read_config_from_form(self) -> SimulationConfig:
        values: dict[str, Any] = {}
        for spec in PARAMETER_SPECS:
            variable = self.variables[spec.name]
            if spec.widget == "check":
                values[spec.name] = bool(variable.get())
            else:
                raw_value = str(variable.get()).strip()
                if spec.name == "random_seed" and raw_value == "":
                    values[spec.name] = None
                    continue
                if spec.parser is int:
                    values[spec.name] = int(raw_value)
                elif spec.parser is float:
                    values[spec.name] = float(raw_value)
                else:
                    values[spec.name] = raw_value
        return SimulationConfig.from_mapping(values)

    def _load_config_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Выберите JSON-конфигурацию",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not file_path:
            return

        try:
            config = SimulationConfig.from_json(Path(file_path))
        except Exception as exc:
            messagebox.showerror("Ошибка загрузки", f"Не удалось прочитать конфигурацию:\n{exc}")
            return

        self.current_config = config
        self._apply_config_to_form(config)
        self.status_var.set(f"Загружена конфигурация: {file_path}")

    def _reset_defaults(self) -> None:
        self.current_config = self._load_initial_config()
        self._apply_config_to_form(self.current_config)
        self.status_var.set("Параметры сброшены к значениям по умолчанию")

    def _start_run(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return

        try:
            config = self._read_config_from_form()
        except Exception as exc:
            messagebox.showerror("Ошибка параметров", f"Не удалось прочитать параметры:\n{exc}")
            return

        self.current_config = config
        self.status_var.set("Идет расчет, пожалуйста подождите...")
        if self.run_button is not None:
            self.run_button.configure(state="disabled")
        if self.save_button is not None:
            self.save_button.configure(state="disabled")

        self.worker_thread = threading.Thread(target=self._run_worker, args=(config,), daemon=True)
        self.worker_thread.start()
        self.root.after(120, self._poll_worker)

    def _run_worker(self, config: SimulationConfig) -> None:
        try:
            result = run_simulation(config)
        except Exception:
            self.result_queue.put(("error", traceback.format_exc()))
            return
        self.result_queue.put(("success", (config, result)))

    def _poll_worker(self) -> None:
        try:
            status, payload = self.result_queue.get_nowait()
        except queue.Empty:
            if self.worker_thread and self.worker_thread.is_alive():
                self.root.after(120, self._poll_worker)
            return

        if self.run_button is not None:
            self.run_button.configure(state="normal")

        if status == "error":
            self.status_var.set("Расчет завершился с ошибкой")
            messagebox.showerror("Ошибка моделирования", str(payload))
            return

        config, results = payload
        self.current_config = config
        self.current_results = results
        if self.save_button is not None:
            self.save_button.configure(state="normal")
        self._update_summary(config, results)
        self._update_plots(config, results)
        self.status_var.set("Расчет завершен")

    def _update_summary(self, config: SimulationConfig, results: SimulationResults) -> None:
        valid_points = int(results.geometric_valid.sum())
        total_points = int(results.distances.size)
        lines = [
            f"Сетка расстояний: {config.L_min:.3f} .. {config.L_max:.3f} м, шаг {config.dL:.3f} м",
            f"Испытания Монте-Карло: N = {config.N}, M = {config.M}, seed = {config.random_seed}",
            f"sigma_w(L) = {config.sigma_w_value:.6f} + {config.sigma_w_slope:.6f} * L",
            f"Точек после геометрической фильтрации: {valid_points} из {total_points}",
            f"Максимальная геометрическая дальность: {format_optional_distance(results.max_geometric_distance)}",
            f"Рабочая дальность при p >= {config.p_required:.3f}: {format_optional_distance(results.operating_distance)}",
        ]
        self.summary_var.set("\n".join(lines))

    def _update_plots(self, config: SimulationConfig, results: SimulationResults) -> None:
        self.ax_probability.clear()
        self.ax_spot.clear()

        if results.max_geometric_distance is not None:
            self.ax_probability.axvspan(
                results.max_geometric_distance,
                float(results.distances[-1]),
                color="#f2dede",
                alpha=0.35,
                label="Отклонено геометрией",
            )

        self.ax_probability.plot(
            results.distances,
            results.success_probabilities,
            color="#2f6b2f",
            linewidth=2.2,
            label="p(L)",
        )
        self.ax_probability.axhline(
            config.p_required,
            color="#b22222",
            linestyle="--",
            linewidth=1.4,
            label="p_required",
        )
        if results.operating_distance is not None:
            self.ax_probability.axvline(
                results.operating_distance,
                color="#0f4c81",
                linestyle=":",
                linewidth=1.4,
                label="L_work",
            )
        self.ax_probability.set_title("Вероятность успеха в зависимости от расстояния")
        self.ax_probability.set_xlabel("Расстояние L, м")
        self.ax_probability.set_ylabel("Вероятность успеха p(L)")
        self.ax_probability.set_ylim(-0.02, 1.02)
        self.ax_probability.grid(True, alpha=0.3)
        self.ax_probability.legend(loc="best")

        self.ax_spot.plot(
            results.distances,
            results.spot_diameters,
            color="#1f5aa6",
            linewidth=2.0,
            label="d(L)",
        )
        self.ax_spot.axhline(
            config.d_target,
            color="#b22222",
            linestyle="--",
            linewidth=1.4,
            label="d_target",
        )
        if results.max_geometric_distance is not None:
            self.ax_spot.axvline(
                results.max_geometric_distance,
                color="#6a3d9a",
                linestyle=":",
                linewidth=1.4,
                label="L_geom_max",
            )
        self.ax_spot.set_title("Диаметр пятна в зависимости от расстояния")
        self.ax_spot.set_xlabel("Расстояние L, м")
        self.ax_spot.set_ylabel("Диаметр пятна d(L), м")
        self.ax_spot.grid(True, alpha=0.3)
        self.ax_spot.legend(loc="best")

        self.figure.tight_layout(pad=2.0)
        if self.canvas is not None:
            self.canvas.draw_idle()

    def _draw_placeholder(self) -> None:
        self.ax_probability.clear()
        self.ax_spot.clear()

        self.ax_probability.set_title("Вероятность успеха p(L)")
        self.ax_probability.set_xlabel("Расстояние L, м")
        self.ax_probability.set_ylabel("p(L)")
        self.ax_probability.set_ylim(-0.02, 1.02)
        self.ax_probability.grid(True, alpha=0.3)
        self.ax_probability.text(
            0.5,
            0.5,
            "Нажмите «Запустить моделирование»",
            ha="center",
            va="center",
            transform=self.ax_probability.transAxes,
        )

        self.ax_spot.set_title("Диаметр пятна d(L)")
        self.ax_spot.set_xlabel("Расстояние L, м")
        self.ax_spot.set_ylabel("d(L), м")
        self.ax_spot.grid(True, alpha=0.3)
        self.ax_spot.text(
            0.5,
            0.5,
            "После расчета здесь появится график",
            ha="center",
            va="center",
            transform=self.ax_spot.transAxes,
        )

        self.figure.tight_layout(pad=2.0)
        if self.canvas is not None:
            self.canvas.draw_idle()

    def _save_results(self) -> None:
        if self.current_results is None:
            return

        output_dir = create_run_output_dir(Path("outputs"))
        artifacts = export_results(self.current_results, self.current_config, output_dir, save_plots=True)
        artifact_lines = [f"{name}: {path}" for name, path in artifacts.items()]
        messagebox.showinfo(
            "Результаты сохранены",
            "Результаты сохранены в:\n"
            f"{output_dir}\n\n"
            + "\n".join(artifact_lines),
        )
        self.status_var.set(f"Результаты сохранены в {output_dir}")


def main() -> None:
    root = tk.Tk()
    try:
        with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            json.load(handle)
    except Exception:
        pass
    app = LaserSimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
