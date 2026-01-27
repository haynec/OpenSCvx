"""Interactive real-time visualization for Dubins car path planning.

This module provides a PyQt5-based GUI for interactively solving and visualizing
the Dubins car trajectory optimization problem in real-time.
"""

import os
import sys
import threading

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsEllipseItem,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from car.dubins_car import (
    # --- Import your problem setup ---
    plotting_dict,
    problem,
)

# --- Shared state for plotting ---
running = {"stop": False}
reset_requested = {"reset": False}
latest_results = {"results": None}
new_result_event = threading.Event()


# --- Key press handler for obstacle movement ---
def on_key(event):
    step = 0.1
    if event.key == "up":
        problem.parameters["obs_center"][1] += step
    elif event.key == "down":
        problem.parameters["obs_center"][1] -= step
    elif event.key == "left":
        problem.parameters["obs_center"][0] -= step
    elif event.key == "right":
        problem.parameters["obs_center"][0] += step
    elif event.key == "escape":
        running["stop"] = True


# --- Optimization loop to run in background thread ---
def optimization_loop():
    problem.initialize()
    print("Optimization loop started...")
    iteration = 0
    try:
        while not running["stop"]:
            # Check if reset was requested
            if reset_requested["reset"]:
                problem.reset()
                reset_requested["reset"] = False
                iteration = 0
                print("Problem reset to initial conditions")

            # Perform a single SCP step (automatically warm-starts from previous iteration)
            print(f"Starting iteration {iteration}...")
            step_result = problem.step()
            iteration += 1

            # Build results dict for visualization
            results = {
                "iter": step_result["scp_k"] - 1,  # Display iteration (0-indexed)
                "J_tr": step_result["scp_J_tr"],
                "J_vb": step_result["scp_J_vb"],
                "J_vc": step_result["scp_J_vc"],
                "converged": step_result["converged"],
                "V_multi_shoot": problem.state.V_history[-1] if problem.state.V_history else [],
                "x": problem.state.x,  # Current state trajectory
                "u": problem.state.u,  # Current control trajectory
            }

            # Get timing from the print queue (emitted data)
            try:
                if hasattr(problem, "print_queue") and not problem.print_queue.empty():
                    # Get the latest emitted data
                    emitted_data = problem.print_queue.get_nowait()
                    results["dis_time"] = emitted_data.get("dis_time", 0.0)
                    results["solve_time"] = emitted_data.get("subprop_time", 0.0)
                    results["prob_stat"] = emitted_data.get("prob_stat", "--")
                    results["cost"] = emitted_data.get("cost", 0.0)
                else:
                    results["dis_time"] = 0.0
                    results["solve_time"] = 0.0
                    results["prob_stat"] = "--"
                    results["cost"] = 0.0
            except Exception:
                results["dis_time"] = 0.0
                results["solve_time"] = 0.0
                results["prob_stat"] = "--"
                results["cost"] = 0.0

            # Print iteration info to CLI
            print(
                f"Iteration {iteration}: J_tr={results['J_tr']:.2e}, J_vb={results['J_vb']:.2e}, "
                f"J_vc={results['J_vc']:.2e}, Cost={results['cost']:.2e}, "
                f"Status={results['prob_stat']}"
            )

            results.update(plotting_dict)
            latest_results["results"] = results
            new_result_event.set()
            # Check for convergence to optionally stop
            # if results['converged']:
            #     print("Converged!")
            #     # maybe sleep or stop here
    except KeyboardInterrupt:
        running["stop"] = True
        print("Stopped by user.")
    except Exception as e:
        print(f"Error in optimization loop: {e}")
        import traceback

        traceback.print_exc()
        running["stop"] = True


class ObstaclePlotWidget(pg.PlotWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dragging = False

    def keyPressEvent(self, event):
        step = 0.1
        if event.key() == Qt.Key_Up:
            problem.parameters["obs_center"][1] += step
        elif event.key() == Qt.Key_Down:
            problem.parameters["obs_center"][1] -= step
        elif event.key() == Qt.Key_Left:
            problem.parameters["obs_center"][0] -= step
        elif event.key() == Qt.Key_Right:
            problem.parameters["obs_center"][0] += step
        elif event.key() == Qt.Key_Escape:
            running["stop"] = True
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        pos = self.plotItem.vb.mapSceneToView(event.pos())
        mouse_x, mouse_y = pos.x(), pos.y()
        dx = mouse_x - problem.parameters["obs_center"][0]
        dy = mouse_y - problem.parameters["obs_center"][1]
        if dx**2 + dy**2 <= problem.parameters["obs_radius"] ** 2:
            self.dragging = True
            # Do NOT call super() if starting drag (prevents plot pan)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging:
            pos = self.plotItem.vb.mapSceneToView(event.pos())
            problem.parameters["obs_center"][0] = pos.x()
            problem.parameters["obs_center"][1] = pos.y()
            # Do NOT call super() if dragging (prevents plot pan)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.dragging:
            self.dragging = False
            # Do NOT call super() if ending drag (prevents plot pan)
        else:
            super().mouseReleaseEvent(event)


def on_lam_cost_changed(input_widget):
    """Handle lambda cost input changes"""
    # Extract the new value from the input widget
    new_value = input_widget.text()
    try:
        # Convert the new value to a float
        lam_cost_value = float(new_value)
        problem.settings.scp.lam_cost = lam_cost_value
        # Update the display with scientific notation
        input_widget.setText(f"{lam_cost_value:.2E}")
    except ValueError:
        print("Invalid input. Please enter a valid number.")


def on_lam_tr_changed(input_widget):
    """Handle lambda trust region input changes"""
    # Extract the new value from the input widget
    new_value = input_widget.text()
    try:
        # Convert the new value to a float
        lam_tr_value = float(new_value)
        problem.settings.scp.lam_prox = lam_tr_value
        # Update the display with scientific notation
        input_widget.setText(f"{lam_tr_value:.2E}")
    except ValueError:
        print("Invalid input. Please enter a valid number.")


def on_reset_clicked():
    """Handle reset button click"""
    reset_requested["reset"] = True
    print("Problem reset requested")


def update_optimization_metrics(results, labels_dict):
    """Update the optimization metrics display"""
    if results is None:
        return
    # Extract metrics from results
    iter_num = results.get("iter", 0)
    j_tr = results.get("J_tr", 0.0)
    j_vb = results.get("J_vb", 0.0)
    j_vc = results.get("J_vc", 0.0)
    cost = results.get("cost", 0.0)
    status = results.get("prob_stat", "--")
    # Get timing information (these would need to be tracked separately)
    dis_time = results.get("dis_time", 0.0)
    solve_time = results.get("solve_time", 0.0)
    # Update labels
    labels_dict["iter_label"].setText(f"Iteration: {iter_num}")
    labels_dict["j_tr_label"].setText(f"J_tr: {j_tr:.2E}")
    labels_dict["j_vb_label"].setText(f"J_vb: {j_vb:.2E}")
    labels_dict["j_vc_label"].setText(f"J_vc: {j_vc:.2E}")
    labels_dict["objective_label"].setText(f"Objective: {cost:.2E}")
    labels_dict["lam_cost_display_label"].setText(f"λ_cost: {problem.settings.scp.lam_cost:.2E}")
    labels_dict["dis_time_label"].setText(f"Dis Time: {dis_time:.1f}ms")
    labels_dict["solve_time_label"].setText(f"Solve Time: {solve_time:.1f}ms")
    labels_dict["status_label"].setText(f"Status: {status}")


def plot_thread_func():
    # Initialize PyQtGraph
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    # Create main window
    main_widget = QWidget()
    main_widget.setWindowTitle("Dubins Car Real-time Trajectory")
    main_layout = QHBoxLayout()
    main_widget.setLayout(main_layout)
    # Create plot window using the custom widget
    plot_widget = ObstaclePlotWidget()
    plot_widget.setLabel("left", "Y Position")
    plot_widget.setLabel("bottom", "X Position")
    # Create control panel
    control_panel = QWidget()
    control_layout = QVBoxLayout()
    control_panel.setLayout(control_layout)
    # Title
    title = QLabel("Dubins Car Control")
    title.setStyleSheet("font-weight: bold; font-size: 14px;")
    control_layout.addWidget(title)
    # Optimization Metrics Display
    metrics_group = QGroupBox("Optimization Metrics")
    metrics_layout = QVBoxLayout()
    metrics_group.setLayout(metrics_layout)
    # Create labels for each metric
    iter_label = QLabel("Iteration: 0")
    j_tr_label = QLabel("J_tr: 0.00E+00")
    j_vb_label = QLabel("J_vb: 0.00E+00")
    j_vc_label = QLabel("J_vc: 0.00E+00")
    objective_label = QLabel("Objective: 0.00E+00")
    lam_cost_display_label = QLabel(f"λ_cost: {problem.settings.scp.lam_cost:.2E}")
    dis_time_label = QLabel("Dis Time: 0.0ms")
    solve_time_label = QLabel("Solve Time: 0.0ms")
    status_label = QLabel("Status: --")
    # Style the labels
    for label in [
        iter_label,
        j_tr_label,
        j_vb_label,
        j_vc_label,
        objective_label,
        lam_cost_display_label,
        dis_time_label,
        solve_time_label,
        status_label,
    ]:
        label.setStyleSheet("font-family: monospace; font-size: 11px; padding: 2px;")
        metrics_layout.addWidget(label)
    # Create labels dictionary for metrics update
    labels_dict = {
        "iter_label": iter_label,
        "j_tr_label": j_tr_label,
        "j_vb_label": j_vb_label,
        "j_vc_label": j_vc_label,
        "objective_label": objective_label,
        "lam_cost_display_label": lam_cost_display_label,
        "dis_time_label": dis_time_label,
        "solve_time_label": solve_time_label,
        "status_label": status_label,
    }
    control_layout.addWidget(metrics_group)
    # Optimization Weights
    weights_group = QGroupBox("Optimization Weights")
    weights_layout = QVBoxLayout()
    weights_group.setLayout(weights_layout)
    # Lambda cost input - Input on left, label on right
    lam_cost_layout = QHBoxLayout()
    lam_cost_input = QLineEdit()
    lam_cost_input.setText(f"{problem.settings.scp.lam_cost:.2E}")
    lam_cost_input.setFixedWidth(80)
    lam_cost_input.returnPressed.connect(lambda: on_lam_cost_changed(lam_cost_input))
    lam_cost_label = QLabel("λ_cost:")
    lam_cost_label.setAlignment(Qt.AlignLeft)
    lam_cost_layout.addWidget(lam_cost_input)
    lam_cost_layout.addWidget(lam_cost_label)
    lam_cost_layout.addStretch()  # Push everything to the left
    weights_layout.addLayout(lam_cost_layout)
    # Lambda trust region input - Input on left, label on right
    lam_tr_layout = QHBoxLayout()
    lam_tr_input = QLineEdit()
    lam_tr_input.setText(f"{problem.settings.scp.lam_prox:.2E}")
    lam_tr_input.setFixedWidth(80)
    lam_tr_input.returnPressed.connect(lambda: on_lam_tr_changed(lam_tr_input))
    lam_tr_label = QLabel("λ_tr:")
    lam_tr_label.setAlignment(Qt.AlignLeft)
    lam_tr_layout.addWidget(lam_tr_input)
    lam_tr_layout.addWidget(lam_tr_label)
    lam_tr_layout.addStretch()  # Push everything to the left
    weights_layout.addLayout(lam_tr_layout)
    control_layout.addWidget(weights_group)
    # Problem Control
    problem_control_group = QGroupBox("Problem Control")
    problem_control_layout = QVBoxLayout()
    problem_control_group.setLayout(problem_control_layout)
    reset_problem_button = QPushButton("Reset Problem")
    reset_problem_button.clicked.connect(lambda: on_reset_clicked())
    problem_control_layout.addWidget(reset_problem_button)
    control_layout.addWidget(problem_control_group)
    control_layout.addStretch()
    # Add widgets to main layout
    main_layout.addWidget(plot_widget, stretch=3)
    main_layout.addWidget(control_panel, stretch=1)
    main_widget.resize(800, 600)
    main_widget.show()
    # Create scatter plot item for trajectory
    traj_scatter = pg.ScatterPlotItem(pen=None, symbol="o", size=5, brush="b")
    plot_widget.addItem(traj_scatter)
    # Create circle for obstacle with true radius
    obs_circle = QGraphicsEllipseItem(
        problem.parameters["obs_center"][0] - problem.parameters["obs_radius"],
        problem.parameters["obs_center"][1] - problem.parameters["obs_radius"],
        problem.parameters["obs_radius"] * 2,
        problem.parameters["obs_radius"] * 2,
    )
    obs_circle.setPen(pg.mkPen("g", width=2))
    obs_circle.setBrush(pg.mkBrush(0, 255, 0, 60))
    plot_widget.addItem(obs_circle)
    # Set initial plot limits
    plot_widget.setXRange(-2, 2)
    plot_widget.setYRange(-2, 2)
    # Update timer
    timer = QTimer()

    def update_plot():
        if latest_results["results"] is not None:
            try:
                V_multi_shoot = np.array(latest_results["results"]["V_multi_shoot"])
                n_x = problem.settings.sim.n_states
                n_u = problem.settings.sim.n_controls
                i1 = n_x
                i2 = i1 + n_x * n_x
                i3 = i2 + n_x * n_u
                i4 = i3 + n_x * n_u
                all_pos_segments = []
                for i_node in range(V_multi_shoot.shape[1]):
                    node_data = V_multi_shoot[:, i_node]
                    segments_for_node = node_data.reshape(-1, i4)
                    pos_segments = segments_for_node[:, :2]
                    all_pos_segments.append(pos_segments)
                if all_pos_segments:
                    full_traj = np.vstack(all_pos_segments)
                    traj_scatter.setData(full_traj[:, 0], full_traj[:, 1])
                # Update obstacle circle position
                obs_circle.setRect(
                    problem.parameters["obs_center"][0] - problem.parameters["obs_radius"],
                    problem.parameters["obs_center"][1] - problem.parameters["obs_radius"],
                    problem.parameters["obs_radius"] * 2,
                    problem.parameters["obs_radius"] * 2,
                )
                # Update optimization metrics display
                update_optimization_metrics(latest_results["results"], labels_dict)
            except Exception as e:
                print(f"Plot update error: {e}")
                if "x" in latest_results["results"]:
                    x_traj = latest_results["results"]["x"]  # Now a numpy array
                    traj_scatter.setData(x_traj[:, 0], x_traj[:, 1])

    timer.timeout.connect(update_plot)
    timer.start(50)
    app.exec_()


if __name__ == "__main__":
    # Start optimization thread
    opt_thread = threading.Thread(target=optimization_loop)
    opt_thread.daemon = True
    opt_thread.start()
    # Start plotting in main thread
    plot_thread_func()
