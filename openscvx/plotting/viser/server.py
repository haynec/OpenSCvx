"""Viser server setup utilities."""

from __future__ import annotations

import os
import socket
import warnings

import matplotlib.pyplot as plt
import numpy as np
import viser
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage


def _localhost_port_available(port: int) -> bool:
    """Return True if nothing is already bound on 127.0.0.1:port.

    Viser defaults to host ``0.0.0.0``, which can still succeed when another
    process (e.g. the Cursor IDE) holds the more-specific ``127.0.0.1`` bind.
    Browsers then hit the other process and the scene never loads.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _resolve_viser_port(port: int | None) -> int:
    """Choose a port that is free on localhost.

    Honors ``_VISER_PORT_OVERRIDE`` when ``port`` is None (used by examples
    that need a fixed secondary port). If the preferred port is already taken
    on ``127.0.0.1``, walk upward so the printed localhost URL actually works.
    """
    preferred = (
        port
        if port is not None
        else int(os.environ.get("_VISER_PORT_OVERRIDE", "8080"))
    )
    if _localhost_port_available(preferred):
        return preferred
    for candidate in range(preferred + 1, preferred + 50):
        if _localhost_port_available(candidate):
            warnings.warn(
                f"Viser port {preferred} is already in use on localhost; "
                f"using {candidate} instead.",
                stacklevel=3,
            )
            return candidate
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def compute_velocity_colors(
    vel: np.ndarray | None,
    cmap_name: str = "viridis",
    fallback_length: int | None = None,
) -> np.ndarray:
    """Compute RGB colors based on velocity magnitude.

    Args:
        vel: Velocity array of shape (N, 3), or None if velocity is not available
        cmap_name: Matplotlib colormap name
        fallback_length: When vel is None, number of points for default color array.
            Required when vel is None.

    Returns:
        Array of RGB colors with shape (N, 3), values in [0, 255]
    """
    if vel is None:
        if fallback_length is None:
            raise ValueError("fallback_length is required when vel is None")
        # Single default color (viridis mid) for all points
        cmap = plt.get_cmap(cmap_name)
        default_rgb = np.array([int(c * 255) for c in cmap(0.5)[:3]], dtype=np.uint8)
        return np.broadcast_to(default_rgb, (fallback_length, 3)).copy()
    vel_norms = np.linalg.norm(vel, axis=1)
    vel_range = vel_norms.max() - vel_norms.min()
    if vel_range < 1e-8:
        vel_normalized = np.zeros_like(vel_norms)
    else:
        vel_normalized = (vel_norms - vel_norms.min()) / vel_range

    cmap = plt.get_cmap(cmap_name)
    colors = np.array([[int(c * 255) for c in cmap(v)[:3]] for v in vel_normalized])
    return colors


def compute_grid_size(
    pos: np.ndarray | None,
    padding: float = 1.2,
    default_size: float = 10.0,
) -> float:
    """Compute grid size based on trajectory extent.

    Args:
        pos: Position array of shape (N, 3), or None if position is not available
        padding: Padding factor (1.2 = 20% padding)
        default_size: Grid size when pos is None

    Returns:
        Grid size (width and height)
    """
    if pos is None:
        return default_size
    max_x = np.abs(pos[:, 0]).max()
    max_y = np.abs(pos[:, 1]).max()
    return max(max_x, max_y) * 2 * padding


def create_server(
    pos: np.ndarray | None,
    dark_mode: bool = True,
    show_grid: bool = True,
    port: int | None = None,
) -> viser.ViserServer:
    """Create a viser server with basic scene setup.

    Args:
        pos: Position array for computing grid size, or None to use default grid size
        dark_mode: Whether to use dark theme
        show_grid: Whether to show the grid (default True)
        port: Preferred HTTP port. Defaults to 8080 (or ``_VISER_PORT_OVERRIDE``).
            If that port is already bound on localhost, the next free port is used.

    Returns:
        ViserServer instance with grid and origin frame
    """
    server = viser.ViserServer(port=_resolve_viser_port(port))

    # Configure theme with OpenSCvx branding
    # TitlebarButton and TitlebarConfig are TypedDict classes (create as plain dicts)
    buttons = (
        TitlebarButton(
            text="API Reference",
            icon="Description",
            href="https://openscvx.github.io/OpenSCvx/latest/Reference/",
        ),
        TitlebarButton(
            text="Docs",
            icon="Description",
            href="https://openscvx.github.io/OpenSCvx/",
        ),
        TitlebarButton(
            text="GitHub",
            icon="GitHub",
            href="https://github.com/OpenSCvx/OpenSCvx",
        ),
    )

    # Add OpenSCvx logo to titlebar (loaded from GitHub)
    logo_url = (
        "https://raw.githubusercontent.com/OpenSCvx/OpenSCvx/main/figures/openscvx_logo_square.png"
    )
    image = TitlebarImage(
        image_url_light=logo_url,
        image_url_dark=logo_url,  # Use same logo for both themes
        image_alt="OpenSCvx",
        href="https://github.com/OpenSCvx/OpenSCvx",
    )

    titlebar_config = TitlebarConfig(buttons=buttons, image=image)

    server.gui.configure_theme(
        titlebar_content=titlebar_config,
        dark_mode=dark_mode,
    )

    if show_grid:
        grid_size = compute_grid_size(pos)
        server.scene.add_grid(
            "/grid",
            width=grid_size,
            height=grid_size,
            position=np.array([0.0, 0.0, 0.0]),
        )
    server.scene.add_frame(
        "/origin",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
    )

    return server
