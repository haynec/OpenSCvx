"""Viser server setup utilities."""

import matplotlib.pyplot as plt
import numpy as np
import viser
from matplotlib.colors import Colormap
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage


def compute_velocity_colors(
    vel: np.ndarray | None,
    cmap_name: str = "viridis",
    fallback_length: int | None = None,
    cmap: Colormap | None = None,
) -> np.ndarray:
    """Compute RGB colors based on velocity magnitude.

    Args:
        vel: Velocity array of shape (N, 3), or None if velocity is not available
        cmap_name: Matplotlib colormap name
        fallback_length: When vel is None, number of points for default color array.
            Required when vel is None.
        cmap: Preloaded colormap to use instead of looking up ``cmap_name``. Pass
            one when recoloring every frame of a realtime loop, where the lookup
            would otherwise repeat per call.

    Returns:
        Array of RGB colors with shape (N, 3), values in [0, 255]
    """
    if cmap is None:
        cmap = plt.get_cmap(cmap_name)

    if vel is None:
        if fallback_length is None:
            raise ValueError("fallback_length is required when vel is None")
        # Single default color (viridis mid) for all points
        default_rgb = np.array([int(c * 255) for c in cmap(0.5)[:3]], dtype=np.uint8)
        return np.broadcast_to(default_rgb, (fallback_length, 3)).copy()
    vel_norms = np.linalg.norm(vel, axis=1)
    vel_range = vel_norms.max() - vel_norms.min()
    if vel_range < 1e-8:
        vel_normalized = np.zeros_like(vel_norms)
    else:
        vel_normalized = (vel_norms - vel_norms.min()) / vel_range

    return (np.asarray(cmap(vel_normalized))[:, :3] * 255).astype(int)


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
    *,
    show_origin: bool = True,
    port: int | None = None,
) -> viser.ViserServer:
    """Create a viser server with basic scene setup.

    Prefer this over a bare ``viser.ViserServer()``: it is the only thing that
    attaches the OpenSCvx titlebar (logo and documentation links), so a scene
    built without it silently ships unbranded.

    Args:
        pos: Position array for computing grid size, or None to use default grid size
        dark_mode: Whether to use dark theme
        show_grid: Whether to show the grid (default True)
        show_origin: Whether to draw the origin triad. The frame is half a metre
            across, so switch it off for scenes whose whole extent is on that
            order (a scale-model track, a tabletop workspace) where it would
            dominate rather than orient.
        port: Port to serve on. Pass distinct ports when one example runs several
            servers at once; ``None`` uses viser's default.

    Returns:
        ViserServer instance with the titlebar, and optionally a grid and origin frame
    """
    server = viser.ViserServer() if port is None else viser.ViserServer(port=port)

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
    if show_origin:
        server.scene.add_frame(
            "/origin",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
        )

    return server
