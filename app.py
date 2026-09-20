import math

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

G = 9.81  # m/s^2

MATERIALS = {
    "Steel": {"E_GPa": 200.0, "density": 7850.0},
    "Aluminium": {"E_GPa": 69.0, "density": 2700.0},
    "Titanium (Ti-6Al-4V)": {"E_GPa": 114.0, "density": 4430.0},
    "Carbon Fibre Composite (quasi-isotropic)": {"E_GPa": 70.0, "density": 1600.0},
    "Custom": None,
}

BEAM_TYPES = {
    "hollow_rect": "Hollow Rectangle Tube",
    "hollow_round": "Hollow Round Tube",
    "solid_round": "Solid Round Bar",
    "solid_rect": "Solid Rectangular Bar",
}

SUPPORT_TYPES = {
    "simply_supported": "Simply Supported (both ends)",
    "cantilever": "Cantilever (fixed left end)",
    "fixed_fixed": "Fixed Supports (both ends)",
}

LOAD_TYPES = {
    "udl": "Evenly Distributed Load",
    "point": "Point Load",
}

st.set_page_config(page_title="Beam Calculator", layout="centered")


# ---------------------------------------------------------------------------
# Icon / diagram drawing helpers
# ---------------------------------------------------------------------------
def _new_icon_ax(figsize=(1.7, 1.7)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_aspect("equal")
    return fig, ax


def fig_hollow_rect():
    fig, ax = _new_icon_ax()
    ax.add_patch(plt.Rectangle((-1, -0.7), 2, 1.4, facecolor="#8a8a8a", edgecolor="black", linewidth=1.5))
    ax.add_patch(plt.Rectangle((-0.65, -0.4), 1.3, 0.8, facecolor="white", edgecolor="black", linewidth=1.5))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.0, 1.0)
    return fig


def fig_hollow_round():
    fig, ax = _new_icon_ax()
    ax.add_patch(plt.Circle((0, 0), 0.9, facecolor="#8a8a8a", edgecolor="black", linewidth=1.5))
    ax.add_patch(plt.Circle((0, 0), 0.55, facecolor="white", edgecolor="black", linewidth=1.5))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    return fig


def fig_solid_round():
    fig, ax = _new_icon_ax()
    ax.add_patch(plt.Circle((0, 0), 0.9, facecolor="#8a8a8a", edgecolor="black", linewidth=1.5))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    return fig


def fig_solid_rect():
    fig, ax = _new_icon_ax()
    ax.add_patch(plt.Rectangle((-1, -0.7), 2, 1.4, facecolor="#8a8a8a", edgecolor="black", linewidth=1.5))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.0, 1.0)
    return fig


CROSS_SECTION_FIGS = {
    "hollow_rect": fig_hollow_rect,
    "hollow_round": fig_hollow_round,
    "solid_round": fig_solid_round,
    "solid_rect": fig_solid_rect,
}


def _hatching(ax, x0, y0, y1, n, direction):
    ax.plot([x0, x0], [y0, y1], color="black", linewidth=3)
    for y in np.linspace(y0, y1, n):
        ax.plot([x0, x0 + 0.25 * direction], [y, y - 0.2], color="black", linewidth=1)


def fig_simply_supported():
    fig, ax = plt.subplots(figsize=(2.6, 1.3))
    ax.axis("off")
    ax.plot([0, 4], [1, 1], color="black", linewidth=3)
    for xpos in (0, 4):
        ax.add_patch(plt.Polygon([[xpos - 0.22, 0.35], [xpos + 0.22, 0.35], [xpos, 1]], closed=True, facecolor="black"))
        ax.plot([xpos - 0.35, xpos + 0.35], [0.32, 0.32], color="black", linewidth=1.5)
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(0.1, 1.5)
    return fig


def fig_cantilever():
    fig, ax = plt.subplots(figsize=(2.6, 1.3))
    ax.axis("off")
    ax.plot([0, 4], [1, 1], color="black", linewidth=3)
    _hatching(ax, 0, 0.4, 1.6, 6, -1)
    ax.set_xlim(-0.8, 4.6)
    ax.set_ylim(0.1, 2.0)
    return fig


def fig_fixed_fixed():
    fig, ax = plt.subplots(figsize=(2.6, 1.3))
    ax.axis("off")
    ax.plot([0, 4], [1, 1], color="black", linewidth=3)
    _hatching(ax, 0, 0.4, 1.6, 6, -1)
    _hatching(ax, 4, 0.4, 1.6, 6, 1)
    ax.set_xlim(-0.8, 4.8)
    ax.set_ylim(0.1, 2.0)
    return fig


SUPPORT_FIGS = {
    "simply_supported": fig_simply_supported,
    "cantilever": fig_cantilever,
    "fixed_fixed": fig_fixed_fixed,
}


def fig_udl():
    fig, ax = plt.subplots(figsize=(2.6, 1.3))
    ax.axis("off")
    ax.plot([0, 4], [1, 1], color="black", linewidth=3)
    for x in np.linspace(0.2, 3.8, 8):
        ax.annotate("", xy=(x, 1.02), xytext=(x, 1.6), arrowprops=dict(arrowstyle="->", linewidth=1.3))
    ax.plot([0, 4], [1.6, 1.6], color="black", linewidth=1)
    ax.set_xlim(-0.4, 4.4)
    ax.set_ylim(0.5, 2.0)
    return fig


def fig_point_load():
    fig, ax = plt.subplots(figsize=(2.6, 1.3))
    ax.axis("off")
    ax.plot([0, 4], [1, 1], color="black", linewidth=3)
    ax.annotate("", xy=(2, 1.02), xytext=(2, 1.9), arrowprops=dict(arrowstyle="->", linewidth=2.2))
    ax.text(2, 1.95, "P", ha="center", fontsize=11)
    ax.set_xlim(-0.4, 4.4)
    ax.set_ylim(0.5, 2.2)
    return fig


LOAD_FIGS = {
    "udl": fig_udl,
    "point": fig_point_load,
}


def render_choice_row(items, state_key, fig_lookup, label_lookup, cols_ratio=None):
    """Render a row of icon + button choices, storing selection in st.session_state[state_key]."""
    cols = st.columns(len(items))
    for col, key in zip(cols, items):
        with col:
            fig = fig_lookup[key]()
            st.pyplot(fig)
            plt.close(fig)
            is_selected = st.session_state.get(state_key) == key
            if st.button(
                label_lookup[key],
                key=f"btn_{state_key}_{key}",
                type="primary" if is_selected else "secondary",
                width="stretch",
            ):
                if st.session_state[state_key] != key:
                    st.session_state[state_key] = key
                    st.rerun()
            if is_selected:
                st.caption("✓ Selected")


# ---------------------------------------------------------------------------
# Section properties
# ---------------------------------------------------------------------------
def section_properties(beam_type, dims_m):
    """dims_m: dict of dimensions already converted to metres. Returns (area_m2, I_m4, c_m)."""
    if beam_type == "hollow_rect":
        h, w, t = dims_m["h"], dims_m["w"], dims_m["t"]
        hi, wi = h - 2 * t, w - 2 * t
        area = h * w - hi * wi
        i_val = (w * h ** 3 - wi * hi ** 3) / 12
        c = h / 2
    elif beam_type == "hollow_round":
        od, t = dims_m["OD"], dims_m["t"]
        id_ = od - 2 * t
        area = math.pi / 4 * (od ** 2 - id_ ** 2)
        i_val = math.pi / 64 * (od ** 4 - id_ ** 4)
        c = od / 2
    elif beam_type == "solid_round":
        od = dims_m["OD"]
        area = math.pi / 4 * od ** 2
        i_val = math.pi / 64 * od ** 4
        c = od / 2
    elif beam_type == "solid_rect":
        h, w = dims_m["h"], dims_m["w"]
        area = h * w
        i_val = w * h ** 3 / 12
        c = h / 2
    else:
        raise ValueError(beam_type)
    return area, i_val, c


# ---------------------------------------------------------------------------
# Beam deflection / moment formulas (x, L in metres; w in N/m; P in N)
# ---------------------------------------------------------------------------
def udl_deflection(support, w, x, L, E, I):
    if support == "simply_supported":
        return (w * x / (24 * E * I)) * (L ** 3 - 2 * L * x ** 2 + x ** 3)
    if support == "cantilever":
        return (w * x ** 2 * (6 * L ** 2 - 4 * L * x + x ** 2)) / (24 * E * I)
    if support == "fixed_fixed":
        return (w * x ** 2 * (L - x) ** 2) / (24 * E * I)
    raise ValueError(support)


def udl_moment(support, w, x, L):
    if support == "simply_supported":
        return w * x * (L - x) / 2
    if support == "cantilever":
        return w * (L - x) ** 2 / 2
    if support == "fixed_fixed":
        return -(w * (6 * x ** 2 - 6 * L * x + L ** 2) / 12)
    raise ValueError(support)


def point_deflection(support, P, a, x, L, E, I):
    b = L - a
    if support == "simply_supported":
        return np.where(
            x <= a,
            (P * b * x) / (6 * L * E * I) * (L ** 2 - b ** 2 - x ** 2),
            (P * a * (L - x)) / (6 * L * E * I) * (2 * L * x - a ** 2 - x ** 2),
        )
    if support == "cantilever":
        return np.where(
            x <= a,
            P * x ** 2 * (3 * a - x) / (6 * E * I),
            P * a ** 2 * (3 * x - a) / (6 * E * I),
        )
    if support == "fixed_fixed":
        return np.where(
            x <= a,
            (P * b ** 2 * x ** 2) / (6 * E * I * L ** 3) * (3 * a * L - 3 * a * x - b * x),
            (P * a ** 2 * (L - x) ** 2) / (6 * E * I * L ** 3) * (3 * b * L - 3 * b * (L - x) - a * (L - x)),
        )
    raise ValueError(support)


def point_moment(support, P, a, x, L):
    b = L - a
    if support == "simply_supported":
        return np.where(x <= a, P * b * x / L, P * a * (L - x) / L)
    if support == "cantilever":
        return np.where(x <= a, P * (a - x), 0.0)
    if support == "fixed_fixed":
        Ma = P * a * b ** 2 / L ** 2
        Ra = P * b ** 2 * (L + 2 * a) / L ** 3
        return np.where(x <= a, Ra * x - Ma, Ra * x - Ma - P * (x - a))
    raise ValueError(support)


# ---------------------------------------------------------------------------
# App state defaults
# ---------------------------------------------------------------------------
st.session_state.setdefault("beam_type", "hollow_rect")
st.session_state.setdefault("support_type", "simply_supported")
st.session_state.setdefault("load_type", "udl")

st.title("Beam Calculator")
st.caption("Maximum deflection and stress for a simple beam, including the beam's own weight.")

# --- Step 1: cross-section type -------------------------------------------------
st.header("1. Beam cross-section")
render_choice_row(list(BEAM_TYPES.keys()), "beam_type", CROSS_SECTION_FIGS, BEAM_TYPES)
beam_type = st.session_state["beam_type"]

st.subheader("Dimensions (mm)")
dims_mm = {}
if beam_type == "hollow_rect":
    c1, c2, c3 = st.columns(3)
    dims_mm["h"] = c1.number_input("Height", min_value=0.1, value=50.0, step=1.0)
    dims_mm["w"] = c2.number_input("Width", min_value=0.1, value=30.0, step=1.0)
    dims_mm["t"] = c3.number_input("Wall thickness", min_value=0.01, value=3.0, step=0.5)
elif beam_type == "hollow_round":
    c1, c2 = st.columns(2)
    dims_mm["OD"] = c1.number_input("Outer diameter (OD)", min_value=0.1, value=50.0, step=1.0)
    dims_mm["t"] = c2.number_input("Wall thickness", min_value=0.01, value=3.0, step=0.5)
elif beam_type == "solid_round":
    dims_mm["OD"] = st.number_input("Outer diameter (OD)", min_value=0.1, value=30.0, step=1.0)
elif beam_type == "solid_rect":
    c1, c2 = st.columns(2)
    dims_mm["h"] = c1.number_input("Height", min_value=0.1, value=50.0, step=1.0)
    dims_mm["w"] = c2.number_input("Width", min_value=0.1, value=30.0, step=1.0)

dims_error = None
if beam_type in ("hollow_rect",) and dims_mm["t"] * 2 >= min(dims_mm["h"], dims_mm["w"]):
    dims_error = "Wall thickness is too large for the given height/width."
elif beam_type in ("hollow_round",) and dims_mm["t"] * 2 >= dims_mm["OD"]:
    dims_error = "Wall thickness is too large for the given outer diameter."

if dims_error:
    st.error(dims_error)

# --- Step 2: support type --------------------------------------------------------
st.header("2. Support type")
render_choice_row(list(SUPPORT_TYPES.keys()), "support_type", SUPPORT_FIGS, SUPPORT_TYPES)
support_type = st.session_state["support_type"]

length_m = st.number_input("Beam length (m)", min_value=0.01, value=1.0, step=0.1)

# --- Step 3: load type -----------------------------------------------------------
st.header("3. Load type")
render_choice_row(list(LOAD_TYPES.keys()), "load_type", LOAD_FIGS, LOAD_TYPES)
load_type = st.session_state["load_type"]

load_error = None
if load_type == "udl":
    total_applied_load_n = st.number_input("Total distributed load (N)", min_value=0.0, value=500.0, step=10.0)
    point_load_n = None
    point_a_m = None
else:
    c1, c2 = st.columns(2)
    point_load_n = c1.number_input("Point load (N)", min_value=0.0, value=500.0, step=10.0)
    point_a_m = c2.number_input("Distance from left support (m)", min_value=0.0, value=length_m / 2, step=0.05)
    total_applied_load_n = None
    if point_a_m >= length_m or point_a_m < 0:
        load_error = "Load distance must be between 0 and the beam length."

if load_error:
    st.error(load_error)

# --- Step 4: material --------------------------------------------------------------
st.header("4. Material")
material_name = st.selectbox("Material", list(MATERIALS.keys()))
if material_name == "Custom":
    c1, c2 = st.columns(2)
    E_GPa = c1.number_input("Elastic modulus E (GPa)", min_value=0.001, value=200.0, step=1.0)
    density = c2.number_input("Density (kg/m³)", min_value=0.001, value=7850.0, step=10.0)
else:
    preset = MATERIALS[material_name]
    E_GPa, density = preset["E_GPa"], preset["density"]
    c1, c2 = st.columns(2)
    c1.info(f"E = {E_GPa} GPa")
    c2.info(f"Density = {density} kg/m³")

# ---------------------------------------------------------------------------
# Calculation
# ---------------------------------------------------------------------------
st.header("Results")

if dims_error or load_error:
    st.warning("Fix the input errors above to see results.")
else:
    dims_m = {k: v / 1000.0 for k, v in dims_mm.items()}
    area_m2, I_m4, c_m = section_properties(beam_type, dims_m)

    E_pa = E_GPa * 1e9
    mass_kg = density * area_m2 * length_m
    w_self = density * area_m2 * G  # N/m

    x = np.linspace(0, length_m, 2001)

    if load_type == "udl":
        w_total = w_self + (total_applied_load_n / length_m)
        y = udl_deflection(support_type, w_total, x, length_m, E_pa, I_m4)
        m = udl_moment(support_type, w_total, x, length_m)
    else:
        y = udl_deflection(support_type, w_self, x, length_m, E_pa, I_m4)
        m = udl_moment(support_type, w_self, x, length_m)
        y = y + point_deflection(support_type, point_load_n, point_a_m, x, length_m, E_pa, I_m4)
        m = m + point_moment(support_type, point_load_n, point_a_m, x, length_m)

    max_deflection_m = float(np.max(np.abs(y)))
    max_moment_nm = float(np.max(np.abs(m)))
    max_stress_pa = max_moment_nm * c_m / I_m4

    c1, c2, c3 = st.columns(3)
    c1.metric("Max deflection", f"{max_deflection_m * 1000:.3f} mm")
    c2.metric("Max stress", f"{max_stress_pa / 1e6:.2f} MPa")
    c3.metric("Beam mass", f"{mass_kg:.3f} kg")

    with st.expander("Section properties"):
        st.write(f"Cross-sectional area: {area_m2 * 1e6:.2f} mm²")
        st.write(f"Second moment of area (I): {I_m4 * 1e12:.2f} mm⁴")
        st.write(f"Distance to extreme fibre (c): {c_m * 1000:.2f} mm")
        st.write(f"Self-weight distributed load: {w_self:.3f} N/m")

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(x, -y * 1000, color="#1f77b4")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Position along beam (m)")
    ax.set_ylabel("Deflection (mm)")
    ax.set_title("Deflected shape")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
