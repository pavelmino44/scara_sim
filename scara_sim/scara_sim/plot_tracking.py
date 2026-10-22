#!/usr/bin/env python3
"""
Графики качества слежения по логам pid_controller.

Можно передать несколько логов (например, режимы ff, pid, ff+pid)
- они будут наложены друг на друга для сравнения.

    ros2 run scara_sim plot_tracking log_ff.csv log_pid.csv log_ff_pid.csv
    ros2 run scara_sim plot_tracking --latest           # последний лог из data/
    ros2 run scara_sim plot_tracking a.csv --save out/  # сохранить PNG
"""

import argparse
import csv
import glob
import os

import numpy as np
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        data = np.array([[float(v) for v in row] for row in r if row])
    return {h: data[:, i] for i, h in enumerate(header)}


def label_of(path):
    name = os.path.basename(path)
    act = 'ДПТ ' if '_motor_' in name else ('момент ' if '_torque_' in name else '')
    for tag in ('ff_pid', 'pid', 'ff'):
        if f'_{tag}_' in name:
            return act + tag.replace('_', '+')
    return name


def main():
    p = argparse.ArgumentParser(description='Plot SCARA tracking logs')
    p.add_argument('logs', nargs='*')
    p.add_argument('--latest', action='store_true',
                   help='взять последний лог из share/scara_sim/data')
    p.add_argument('--save', default='', help='папка для PNG (без показа окон)')
    a = p.parse_args()

    logs = list(a.logs)
    if a.latest or not logs:
        from ament_index_python.packages import get_package_share_directory
        files = sorted(glob.glob(os.path.join(
            get_package_share_directory('scara_sim'), 'data', 'pid_log_*.csv')),
            key=os.path.getmtime)
        if not files:
            raise SystemExit('No pid_log_*.csv found')
        logs = [files[-1]]

    runs = [(label_of(f), load(f)) for f in logs]
    n = len([k for k in runs[0][1] if k.startswith('e') and k[1:].isdigit()])

    if a.save:
        import matplotlib
        matplotlib.use('Agg')
        os.makedirs(a.save, exist_ok=True)

    figs = {}

    # 1. Траектория в плоскости XY
    fig, ax = plt.subplots(figsize=(6, 6))
    d0 = runs[0][1]
    ax.plot(d0['x_des'], d0['y_des'], 'k--', lw=2, label='заданная')
    for lab, d in runs:
        ax.plot(d['x'], d['y'], lw=1.2, label=lab)
    ax.set_xlabel('X, м')
    ax.set_ylabel('Y, м')
    ax.set_title('Траектория схвата')
    ax.axis('equal')
    ax.grid(True)
    ax.legend()
    figs['xy'] = fig

    # 2. Декартова ошибка
    fig, ax = plt.subplots(figsize=(9, 4))
    for lab, d in runs:
        e = 1000 * np.hypot(d['x_des'] - d['x'], d['y_des'] - d['y'])
        ax.plot(d['t'], e, label=f'{lab}: RMS {np.sqrt(np.mean(e**2)):.3f} мм')
    rms = [np.sqrt(np.mean(np.hypot(d['x_des'] - d['x'], d['y_des'] - d['y'])**2))
           for _, d in runs]
    if len(runs) > 1 and max(rms) > 30 * max(min(rms), 1e-12):
        ax.set_yscale('log')   # ошибки отличаются на порядки
    ax.set_xlabel('t, с')
    ax.set_ylabel('|e_xy|, мм')
    ax.set_title('Ошибка положения схвата')
    ax.grid(True)
    ax.legend()
    figs['err_xy'] = fig

    # 3. Ошибки по сочленениям
    fig, axs = plt.subplots(n, 1, figsize=(9, 2.6 * n), sharex=True)
    axs = np.atleast_1d(axs)
    for i in range(n):
        for lab, d in runs:
            axs[i].plot(d['t'], 1000 * d[f'e{i + 1}'], label=lab)
        axs[i].set_ylabel(f'e{i + 1}, мрад')
        axs[i].grid(True)
    axs[0].legend()
    axs[0].set_title('Ошибки по углам сочленений')
    axs[-1].set_xlabel('t, с')
    figs['err_q'] = fig

    # 4. Выход регулятора и привод (по каждому логу)
    for lab, d in runs:
        motor = 'i1' in d and not np.all(np.isnan(d['i1']))
        unit = 'В' if motor else 'Н·м'
        sym = 'U' if motor else 'τ'
        rows = 3 if motor else 1
        fig, axs = plt.subplots(n, rows, figsize=(5 * rows, 2.6 * n), sharex=True,
                                squeeze=False)
        for k in range(n):
            ax = axs[k, 0]
            ax.plot(d['t'], d[f'u{k + 1}_ff'], label=f'{sym}_ff')
            ax.plot(d['t'], d[f'u{k + 1}_pid'], label=f'{sym}_pid')
            ax.plot(d['t'], d[f'u{k + 1}'], 'k', lw=0.8, label=sym)
            ax.set_ylabel(f'{sym}{k + 1}, {unit}')
            ax.grid(True)
            if motor:
                axs[k, 1].plot(d['t'], d[f'i{k + 1}'], 'C3')
                axs[k, 1].set_ylabel(f'i{k + 1}, А')
                axs[k, 1].grid(True)
                axs[k, 2].plot(d['t'], d[f'tau{k + 1}'], 'C2')
                axs[k, 2].set_ylabel(f'τ{k + 1} на шарнире, Н·м')
                axs[k, 2].grid(True)
        axs[0, 0].legend()
        axs[0, 0].set_title(f'Выход регулятора ({lab})')
        if motor:
            axs[0, 1].set_title('Ток якоря')
            axs[0, 2].set_title('Момент на шарнире')
        for ax in axs[-1]:
            ax.set_xlabel('t, с')
        safe = lab.replace('ДПТ', 'motor').replace('момент', 'torque').replace('+', '_').replace(' ', '_')
        figs[f'actuator_{safe}'] = fig

    if a.save:
        for name, f in figs.items():
            f.tight_layout()
            f.savefig(os.path.join(a.save, f'{name}.png'), dpi=120)
        print(f'Saved {len(figs)} figures to {a.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
