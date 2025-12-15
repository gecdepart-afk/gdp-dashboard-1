import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Circle

from simpeg.potential_fields.gravity import analytics


def compute_gz_mgal(
    rho_background: float,
    rho_sphere: float,
    radius_m: float,
    depth_center_m: float,
    x_max_km: float,
    n_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (x_km, gz_mgal) along a profile at surface (z=0) for a buried sphere.
    Convention: positive anomaly for positive density contrast (geophysical convention).
    """
    # Density contrast (kg/m^3 -> g/cc)
    delta_rho_kgm3 = rho_sphere - rho_background
    delta_rho_gcc = delta_rho_kgm3 / 1000.0

    # Profile (meters)
    x_m = np.linspace(-x_max_km * 1000.0, x_max_km * 1000.0, int(n_obs))
    y_m = np.zeros_like(x_m)
    z_m = np.zeros_like(x_m)

    # Sphere center (meters). In SimPEG convention, depth is negative z
    xc, yc, zc = 0.0, 0.0, -abs(depth_center_m)

    # Your SimPEG build returns a tuple: (gx, gy, gz)
    gx, gy, gz = analytics.GravSphereFreeSpace(
        x_m, y_m, z_m, radius_m, xc, yc, zc, delta_rho_gcc
    )

    # IMPORTANT:
    # In your earlier run, gz was already in mGal-scale (multiplying by 1e5 was too big).
    # We also flip sign so dense body gives positive anomaly (geophysical convention).
    gz_mgal = -gz

    return x_m / 1000.0, gz_mgal


def make_figure(
    x_km: np.ndarray,
    gz_mgal: np.ndarray,
    rho_background: float,
    rho_sphere: float,
    radius_m: float,
    depth_center_m: float,
):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [2, 1]}
    )

    # ---- Top: anomaly profile
    ax1.plot(x_km, gz_mgal, lw=2)
    ax1.axhline(0, lw=1, color="k")
    ax1.set_xlabel("Distance x (km)")
    ax1.set_ylabel("gz (mGal)")
    ax1.set_title("Gravity anomaly of a buried sphere (analytic)")
    ax1.grid(True)

    delta_rho = rho_sphere - rho_background
    ax1.text(
        0.02,
        0.95,
        f"ρ background = {rho_background:.0f} kg/m³\n"
        f"ρ sphere = {rho_sphere:.0f} kg/m³\n"
        f"Δρ = {delta_rho:.0f} kg/m³\n"
        f"Radius R = {radius_m:.0f} m\n"
        f"Depth to center = {depth_center_m:.0f} m\n"
        f"Convention: positive downward (dense body → + anomaly)",
        transform=ax1.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    # ---- Bottom: schematic cross-section
    ax2.axhline(0, color="k", lw=1)
    sphere = Circle(
        (0.0, -depth_center_m / 1000.0),
        radius_m / 1000.0,
        facecolor="lightcoral",
        edgecolor="k",
        alpha=0.7,
    )
    ax2.add_patch(sphere)

    # Depth arrow
    ax2.annotate(
        "",
        xy=(0.0, -depth_center_m / 1000.0),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="<->"),
    )
    ax2.text(
        0.2,
        -depth_center_m / 2000.0,
        f"Depth = {depth_center_m:.0f} m",
        va="center",
    )

    # Radius arrow
    ax2.annotate(
        "",
        xy=(radius_m / 1000.0, -depth_center_m / 1000.0),
        xytext=(0.0, -depth_center_m / 1000.0),
        arrowprops=dict(arrowstyle="<->"),
    )
    ax2.text(
        (radius_m / 2000.0),
        -depth_center_m / 1000.0 - 0.35,
        f"R = {radius_m:.0f} m",
        ha="center",
    )

    ax2.set_aspect("equal")
    ax2.set_xlim(-max(abs(x_km.min()), abs(x_km.max())), max(abs(x_km.min()), abs(x_km.max())))
    ax2.set_ylim(-(depth_center_m / 1000.0 + 2.0), 1.0)
    ax2.set_xlabel("Distance x (km)")
    ax2.set_ylabel("Depth (km)")
    ax2.set_title("Schematic cross-section")
    ax2.grid(True)

    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Gravity sphere forward model", layout="centered")
    st.title("Gravity forward modelling – buried sphere (analytic)")

    st.markdown(
        "Interactive teaching tool: change densities, radius and depth, and see the gravity anomaly."
    )

    with st.sidebar:
        st.header("Model parameters")

        rho_background = st.number_input(
            "Background density ρ₀ (kg/m³)",
            min_value=1500.0,
            max_value=3500.0,
            value=2700.0,
            step=50.0,
        )

        rho_sphere = st.number_input(
            "Sphere density ρs (kg/m³)",
            min_value=1500.0,
            max_value=4000.0,
            value=3200.0,
            step=50.0,
        )

        radius_m = st.slider(
            "Sphere radius R (m)",
            min_value=50,
            max_value=3000,
            value=600,
            step=50,
        )

        depth_center_m = st.slider(
            "Depth to sphere center (m)",
            min_value=100,
            max_value=10000,
            value=1500,
            step=100,
        )

        x_max_km = st.slider(
            "Half profile length (km)",
            min_value=1.0,
            max_value=50.0,
            value=6.0,
            step=0.5,
        )

        n_obs = st.slider(
            "Number of observation points",
            min_value=51,
            max_value=2001,
            value=241,
            step=10,
        )

    if rho_sphere == rho_background:
        st.warning("Δρ = 0 → anomaly should be ~0 everywhere.")

    x_km, gz_mgal = compute_gz_mgal(
        rho_background=rho_background,
        rho_sphere=rho_sphere,
        radius_m=float(radius_m),
        depth_center_m=float(depth_center_m),
        x_max_km=float(x_max_km),
        n_obs=int(n_obs),
    )

    fig = make_figure(
        x_km=x_km,
        gz_mgal=gz_mgal,
        rho_background=rho_background,
        rho_sphere=rho_sphere,
        radius_m=float(radius_m),
        depth_center_m=float(depth_center_m),
    )

    st.pyplot(fig)

    st.subheader("Quick values")
    st.write(
        {
            "Δρ (kg/m³)": float(rho_sphere - rho_background),
            "Peak |gz| (mGal)": float(np.max(np.abs(gz_mgal))),
        }
    )


if __name__ == "__main__":
    main()
