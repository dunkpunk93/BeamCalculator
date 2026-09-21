import math

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

G = 9.81  # m/s^2

TUBE_MATERIALS = {
    "Steel": {"E_GPa": 200.0, "density": 7850.0},
    "Aluminium": {"E_GPa": 69.0, "density": 2700.0},
    "Titanium (Ti-6Al-4V)": {"E_GPa": 114.0, "density": 4430.0},
    "Carbon Fibre Composite (quasi-isotropic)": {"E_GPa": 70.0, "density": 1600.0},
    "Custom": None,
}

WKSF_MATERIALS = {
    "Particle Board": {"E_GPa": 2.5, "density": 650.0},
    "MDF": {"E_GPa": 3.0, "density": 750.0},
    "Multiplex (Birch Plywood)": {"E_GPa": 10.0, "density": 680.0},
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

TUBE_COUNTS = {
    "one": "1 Tube",
    "two": "2 Tubes",
}

BOND_TYPES = {
    "bonded": "Fully Bonded",
    "unbonded": "Unbonded (Resting Only)",
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


def fig_one_tube():
    fig, ax = _new_icon_ax(figsize=(1.9, 1.9))
    ax.add_patch(plt.Rectangle((-1.1, 0.35), 2.2, 0.22, facecolor="#8a8a8a", edgecolor="black", linewidth=1.5))
    tube_w, tube_h, wall = 0.55, 0.55, 0.09
    ax.add_patch(plt.Rectangle((-tube_w / 2, 0.35 - tube_h), tube_w, tube_h, facecolor="#8a8a8a", edgecolor="black", linewidth=1.5))
    ax.add_patch(plt.Rectangle((-tube_w / 2 + wall, 0.35 - tube_h + wall), tube_w - 2 * wall, tube_h - 2 * wall, facecolor="white", edgecolor="black", linewidth=1.2))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.4, 0.75)
    return fig


def fig_two_tube():
    fig, ax = _new_icon_ax(figsize=(1.9, 1.9))
    ax.add_patch(plt.Rectangle((-1.1, 0.35), 2.2, 0.22, facecolor="#8a8a8a", edgecolor="black", linewidth=1.5))
    tube_w, tube_h, wall = 0.45, 0.55, 0.08
    for cx in (-0.55, 0.55):
        ax.add_patch(plt.Rectangle((cx - tube_w / 2, 0.35 - tube_h), tube_w, tube_h, facecolor="#8a8a8a", edgecolor="black", linewidth=1.5))
        ax.add_patch(plt.Rectangle((cx - tube_w / 2 + wall, 0.35 - tube_h + wall), tube_w - 2 * wall, tube_h - 2 * wall, facecolor="white", edgecolor="black", linewidth=1.2))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.4, 0.75)
    return fig


TUBE_COUNT_FIGS = {
    "one": fig_one_tube,
    "two": fig_two_tube,
}


def fig_bonded():
    fig, ax = _new_icon_ax(figsize=(1.9, 1.9))
    ax.add_patch(plt.Rectangle((-1.0, 0.35), 2.0, 0.22, facecolor="#8a8a8a", edgecolor="black", linewidth=1.5))
    tube_w, tube_h = 0.6, 0.5
    ax.add_patch(plt.Rectangle((-tube_w / 2, 0.35 - tube_h), tube_w, tube_h, facecolor="#8a8a8a", edgecolor="black", linewidth=1.5))
    for cx in np.linspace(-tube_w / 2 + 0.08, tube_w / 2 - 0.08, 5):
        ax.plot([cx, cx + 0.08], [0.35, 0.35 - 0.08], color="black", linewidth=1.3)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.3, 0.7)
    return fig


def fig_unbonded():
    fig, ax = _new_icon_ax(figsize=(1.9, 1.9))
    gap = 0.05
    ax.add_patch(plt.Rectangle((-1.0, 0.35 + gap), 2.0, 0.22, facecolor="#8a8a8a", edgecolor="black", linewidth=1.5))
    tube_w, tube_h = 0.6, 0.5
    ax.add_patch(plt.Rectangle((-tube_w / 2, 0.35 - tube_h), tube_w, tube_h, facecolor="#8a8a8a", edgecolor="black", linewidth=1.5))
    ax.plot([-0.9, 0.9], [0.35 + gap / 2, 0.35 + gap / 2], color="black", linestyle="--", linewidth=1)
    ax.annotate("", xy=(-0.35, 0.15), xytext=(0.35, 0.15), arrowprops=dict(arrowstyle="<->", linewidth=1.2))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.3, 0.75)
    return fig


BOND_FIGS = {
    "bonded": fig_bonded,
    "unbonded": fig_unbonded,
}


def render_choice_row(items, state_key, fig_lookup, label_lookup):
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


def material_selector(materials_dict, key_prefix, default_custom_e_gpa=200.0, default_custom_density=7850.0):
    """Renders a material dropdown (+ custom E/density inputs). Returns (E_GPa, density)."""
    material_name = st.selectbox("Material", list(materials_dict.keys()), key=f"{key_prefix}_material_name")
    if material_name == "Custom":
        c1, c2 = st.columns(2)
        e_gpa = c1.number_input(
            "Elastic modulus E (GPa)", min_value=0.001, value=default_custom_e_gpa, step=1.0, key=f"{key_prefix}_custom_e"
        )
        density = c2.number_input(
            "Density (kg/m³)", min_value=0.001, value=default_custom_density, step=10.0, key=f"{key_prefix}_custom_density"
        )
    else:
        preset = materials_dict[material_name]
        e_gpa, density = preset["E_GPa"], preset["density"]
        c1, c2 = st.columns(2)
        c1.info(f"E = {e_gpa} GPa")
        c2.info(f"Density = {density} kg/m³")
    return e_gpa, density


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
# Beam deflection / moment formulas (x, L in metres; w in N/m; P in N; EI in N.m^2)
# ---------------------------------------------------------------------------
def udl_deflection(support, w, x, L, EI):
    if support == "simply_supported":
        return (w * x / (24 * EI)) * (L ** 3 - 2 * L * x ** 2 + x ** 3)
    if support == "cantilever":
        return (w * x ** 2 * (6 * L ** 2 - 4 * L * x + x ** 2)) / (24 * EI)
    if support == "fixed_fixed":
        return (w * x ** 2 * (L - x) ** 2) / (24 * EI)
    raise ValueError(support)


def udl_moment(support, w, x, L):
    if support == "simply_supported":
        return w * x * (L - x) / 2
    if support == "cantilever":
        return w * (L - x) ** 2 / 2
    if support == "fixed_fixed":
        return -(w * (6 * x ** 2 - 6 * L * x + L ** 2) / 12)
    raise ValueError(support)


def point_deflection(support, P, a, x, L, EI):
    b = L - a
    if support == "simply_supported":
        return np.where(
            x <= a,
            (P * b * x) / (6 * L * EI) * (L ** 2 - b ** 2 - x ** 2),
            (P * a * (L - x)) / (6 * L * EI) * (2 * L * x - a ** 2 - x ** 2),
        )
    if support == "cantilever":
        return np.where(
            x <= a,
            P * x ** 2 * (3 * a - x) / (6 * EI),
            P * a ** 2 * (3 * x - a) / (6 * EI),
        )
    if support == "fixed_fixed":
        return np.where(
            x <= a,
            (P * b ** 2 * x ** 2) / (6 * EI * L ** 3) * (3 * a * L - 3 * a * x - b * x),
            (P * a ** 2 * (L - x) ** 2) / (6 * EI * L ** 3) * (3 * b * L - 3 * b * (L - x) - a * (L - x)),
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


def clamp_point_load_distance(a_m, length_m):
    """If the load sits exactly at (or past) the right support, nudge it 1mm inboard for calc purposes."""
    if a_m >= length_m:
        return max(length_m - 0.001, 1e-6)
    return a_m


def render_load_type_and_get_x_grid(state_prefix, length_m):
    """Renders load-type selector + inputs. Returns (load_type, total_udl_n, point_load_n, point_a_m, calc_point_a_m)."""
    render_choice_row(list(LOAD_TYPES.keys()), f"{state_prefix}_load_type", LOAD_FIGS, LOAD_TYPES)
    load_type = st.session_state[f"{state_prefix}_load_type"]

    magnitude_key = f"{state_prefix}_load_magnitude_n"
    dist_key = f"{state_prefix}_point_a_m"

    if load_type == "udl":
        total_udl_n = st.number_input(
            "Total distributed load (N)", min_value=0.0, value=500.0, step=10.0, key=magnitude_key
        )
        point_load_n = None
        point_a_m = None
        calc_point_a_m = None
    else:
        # Clamp any previously-stored distance if the beam length has since shrunk.
        if dist_key in st.session_state and st.session_state[dist_key] > length_m:
            st.session_state[dist_key] = length_m
        c1, c2 = st.columns(2)
        point_load_n = c1.number_input(
            "Point load (N)", min_value=0.0, value=500.0, step=10.0, key=magnitude_key
        )
        point_a_m = c2.number_input(
            "Distance from left support (m)",
            min_value=0.0,
            max_value=length_m,
            value=min(length_m / 2, length_m),
            step=0.05,
            key=dist_key,
        )
        total_udl_n = None
        calc_point_a_m = clamp_point_load_distance(point_a_m, length_m)

    return load_type, total_udl_n, point_load_n, point_a_m, calc_point_a_m


def deflection_plot(x, y):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(x, -y * 1000, color="#1f77b4")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Position along beam (m)")
    ax.set_ylabel("Deflection (mm)")
    ax.set_title("Deflected shape")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tab 1: Simple Tube Bending
# ---------------------------------------------------------------------------
def render_simple_tube_tab():
    st.session_state.setdefault("t1_beam_type", "hollow_rect")
    st.session_state.setdefault("t1_support_type", "simply_supported")
    st.session_state.setdefault("t1_load_type", "udl")

    st.caption("Maximum deflection and stress for a simple beam, including the beam's own weight.")

    st.header("1. Beam cross-section")
    render_choice_row(list(BEAM_TYPES.keys()), "t1_beam_type", CROSS_SECTION_FIGS, BEAM_TYPES)
    beam_type = st.session_state["t1_beam_type"]

    st.subheader("Dimensions (mm)")
    dims_mm = {}
    if beam_type == "hollow_rect":
        c1, c2, c3 = st.columns(3)
        dims_mm["h"] = c1.number_input("Height", min_value=0.1, value=50.0, step=1.0, key="t1_h")
        dims_mm["w"] = c2.number_input("Width", min_value=0.1, value=30.0, step=1.0, key="t1_w")
        dims_mm["t"] = c3.number_input("Wall thickness", min_value=0.01, value=3.0, step=0.5, key="t1_t")
    elif beam_type == "hollow_round":
        c1, c2 = st.columns(2)
        dims_mm["OD"] = c1.number_input("Outer diameter (OD)", min_value=0.1, value=50.0, step=1.0, key="t1_od")
        dims_mm["t"] = c2.number_input("Wall thickness", min_value=0.01, value=3.0, step=0.5, key="t1_t2")
    elif beam_type == "solid_round":
        dims_mm["OD"] = st.number_input("Outer diameter (OD)", min_value=0.1, value=30.0, step=1.0, key="t1_od2")
    elif beam_type == "solid_rect":
        c1, c2 = st.columns(2)
        dims_mm["h"] = c1.number_input("Height", min_value=0.1, value=50.0, step=1.0, key="t1_h2")
        dims_mm["w"] = c2.number_input("Width", min_value=0.1, value=30.0, step=1.0, key="t1_w2")

    dims_error = None
    if beam_type == "hollow_rect" and dims_mm["t"] * 2 >= min(dims_mm["h"], dims_mm["w"]):
        dims_error = "Wall thickness is too large for the given height/width."
    elif beam_type == "hollow_round" and dims_mm["t"] * 2 >= dims_mm["OD"]:
        dims_error = "Wall thickness is too large for the given outer diameter."

    if dims_error:
        st.error(dims_error)

    st.header("2. Support type")
    render_choice_row(list(SUPPORT_TYPES.keys()), "t1_support_type", SUPPORT_FIGS, SUPPORT_TYPES)
    support_type = st.session_state["t1_support_type"]

    length_m = st.number_input("Beam length (m)", min_value=0.01, value=1.0, step=0.1, key="t1_length")

    st.header("3. Load type")
    load_type, total_udl_n, point_load_n, point_a_m, calc_point_a_m = render_load_type_and_get_x_grid(
        "t1", length_m
    )

    st.header("4. Material")
    e_gpa, density = material_selector(TUBE_MATERIALS, "t1")

    st.header("Results")
    if dims_error:
        st.warning("Fix the input errors above to see results.")
        return

    dims_m = {k: v / 1000.0 for k, v in dims_mm.items()}
    area_m2, I_m4, c_m = section_properties(beam_type, dims_m)

    E_pa = e_gpa * 1e9
    EI = E_pa * I_m4
    mass_kg = density * area_m2 * length_m
    w_self = density * area_m2 * G  # N/m

    x = np.linspace(0, length_m, 2001)

    if load_type == "udl":
        w_total = w_self + (total_udl_n / length_m)
        y = udl_deflection(support_type, w_total, x, length_m, EI)
        m = udl_moment(support_type, w_total, x, length_m)
    else:
        y = udl_deflection(support_type, w_self, x, length_m, EI)
        m = udl_moment(support_type, w_self, x, length_m)
        y = y + point_deflection(support_type, point_load_n, calc_point_a_m, x, length_m, EI)
        m = m + point_moment(support_type, point_load_n, calc_point_a_m, x, length_m)

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

    deflection_plot(x, y)


# ---------------------------------------------------------------------------
# Tab 2: Composite WKSF & Tube Bending
# ---------------------------------------------------------------------------
def compute_composite_properties(wksf_thickness_m, ws_width_m, tube_dims_m, n_tubes, E_wksf_pa, E_tube_pa, bonded):
    """Composite beam properties, vertical stacking only (no horizontal offset or twist).

    bonded=True: transformed-section method (reference material = WKSF) - full shear
    connection, single shared neutral axis, plane sections remain plane across the depth.

    bonded=False: WKSF and tube(s) are only resting in contact (no shear transfer). Both
    layers share the same curvature (they deflect together) but each bends about its own
    centroid; total moment splits between them in proportion to each layer's own EI.
    """
    A_wksf, I_wksf_own, _ = section_properties("solid_rect", {"h": wksf_thickness_m, "w": ws_width_m})
    A_tube_single, I_tube_single_own, _ = section_properties("hollow_rect", tube_dims_m)
    tube_height_m = tube_dims_m["h"]

    A_tube_total = n_tubes * A_tube_single
    I_tube_own_total = n_tubes * I_tube_single_own

    y_wksf = wksf_thickness_m / 2
    y_tube = wksf_thickness_m + tube_height_m / 2

    if bonded:
        n_ratio = E_tube_pa / E_wksf_pa
        A_tube_transformed = n_ratio * A_tube_total

        y_na = (A_wksf * y_wksf + A_tube_transformed * y_tube) / (A_wksf + A_tube_transformed)

        I_transformed = (I_wksf_own + A_wksf * (y_wksf - y_na) ** 2) + n_ratio * (
            I_tube_own_total + A_tube_total * (y_tube - y_na) ** 2
        )

        EI_total = E_wksf_pa * I_transformed
        y_ref_wksf = y_na
        y_ref_tube = y_na
    else:
        n_ratio = None
        I_transformed = None
        EI_total = E_wksf_pa * I_wksf_own + E_tube_pa * I_tube_own_total
        # Each layer bends about its own centroid - no shared neutral axis.
        y_ref_wksf = y_wksf
        y_ref_tube = y_tube

    return {
        "A_wksf": A_wksf,
        "A_tube_total": A_tube_total,
        "y_wksf": y_wksf,
        "y_tube": y_tube,
        "y_ref_wksf": y_ref_wksf,
        "y_ref_tube": y_ref_tube,
        "n_ratio": n_ratio,
        "I_transformed": I_transformed,
        "EI_total": EI_total,
        "tube_height_m": tube_height_m,
        "bonded": bonded,
    }


def render_composite_tab():
    st.session_state.setdefault("t2_tube_count", "one")
    st.session_state.setdefault("t2_bonding_type", "bonded")
    st.session_state.setdefault("t2_support_type", "simply_supported")
    st.session_state.setdefault("t2_load_type", "udl")

    st.caption(
        "Composite worksurface (WKSF) + tube frame beam, vertical stacking only "
        "(no horizontal offset or twist)."
    )

    st.header("1. Tube count")
    render_choice_row(list(TUBE_COUNTS.keys()), "t2_tube_count", TUBE_COUNT_FIGS, TUBE_COUNTS)
    tube_count_key = st.session_state["t2_tube_count"]
    n_tubes = 1 if tube_count_key == "one" else 2

    st.header("2. Bonding type")
    render_choice_row(list(BOND_TYPES.keys()), "t2_bonding_type", BOND_FIGS, BOND_TYPES)
    bonded = st.session_state["t2_bonding_type"] == "bonded"
    if bonded:
        st.caption("Full shear connection assumed (e.g. glued/screwed) - the two layers act as one composite section.")
    else:
        st.caption(
            "No shear connection assumed - the WKSF rests on the tube(s) and each bends about its own "
            "centroid. This gives a lower, more conservative stiffness estimate than full bonding."
        )

    st.header("3. Dimensions (mm)")
    c1, c2 = st.columns(2)
    wksf_thickness_mm = c1.number_input("WKSF thickness", min_value=0.1, value=18.0, step=1.0, key="t2_wksf_t")
    ws_width_mm = c2.number_input("Worksurface width", min_value=1.0, value=700.0, step=10.0, key="t2_ws_width")

    st.caption("Tube dimensions (each tube, hollow rectangular section)")
    c1, c2, c3 = st.columns(3)
    tube_w_mm = c1.number_input("Tube width", min_value=0.1, value=40.0, step=1.0, key="t2_tube_w")
    tube_h_mm = c2.number_input("Tube height", min_value=0.1, value=40.0, step=1.0, key="t2_tube_h")
    tube_t_mm = c3.number_input("Tube wall thickness", min_value=0.01, value=2.0, step=0.5, key="t2_tube_t")

    dims_error = None
    if tube_t_mm * 2 >= min(tube_w_mm, tube_h_mm):
        dims_error = "Tube wall thickness is too large for the given tube width/height."
    if dims_error:
        st.error(dims_error)

    st.header("4. Support type")
    render_choice_row(list(SUPPORT_TYPES.keys()), "t2_support_type", SUPPORT_FIGS, SUPPORT_TYPES)
    support_type = st.session_state["t2_support_type"]

    length_m = st.number_input("WKSF length (m)", min_value=0.01, value=1.2, step=0.1, key="t2_length")

    st.header("5. Load type")
    load_type, total_udl_n, point_load_n, point_a_m, calc_point_a_m = render_load_type_and_get_x_grid(
        "t2", length_m
    )

    st.header("6. Material")
    st.subheader("Worksurface (WKSF)")
    e_wksf_gpa, density_wksf = material_selector(WKSF_MATERIALS, "t2_wksf", default_custom_e_gpa=3.0, default_custom_density=700.0)
    st.subheader("Tube")
    e_tube_gpa, density_tube = material_selector(TUBE_MATERIALS, "t2_tube")

    st.header("Results")
    if dims_error:
        st.warning("Fix the input errors above to see results.")
        return

    wksf_thickness_m = wksf_thickness_mm / 1000.0
    ws_width_m = ws_width_mm / 1000.0
    tube_dims_m = {"h": tube_h_mm / 1000.0, "w": tube_w_mm / 1000.0, "t": tube_t_mm / 1000.0}

    E_wksf_pa = e_wksf_gpa * 1e9
    E_tube_pa = e_tube_gpa * 1e9

    props = compute_composite_properties(
        wksf_thickness_m, ws_width_m, tube_dims_m, n_tubes, E_wksf_pa, E_tube_pa, bonded
    )
    EI_total = props["EI_total"]

    mass_kg = (density_wksf * props["A_wksf"] + density_tube * props["A_tube_total"]) * length_m
    w_self = (density_wksf * props["A_wksf"] + density_tube * props["A_tube_total"]) * G  # N/m

    x = np.linspace(0, length_m, 2001)

    if load_type == "udl":
        w_total = w_self + (total_udl_n / length_m)
        y = udl_deflection(support_type, w_total, x, length_m, EI_total)
        m = udl_moment(support_type, w_total, x, length_m)
    else:
        y = udl_deflection(support_type, w_self, x, length_m, EI_total)
        m = udl_moment(support_type, w_self, x, length_m)
        y = y + point_deflection(support_type, point_load_n, calc_point_a_m, x, length_m, EI_total)
        m = m + point_moment(support_type, point_load_n, calc_point_a_m, x, length_m)

    max_deflection_m = float(np.max(np.abs(y)))
    max_moment_nm = float(np.max(np.abs(m)))

    y_top_wksf = 0.0
    y_interface = wksf_thickness_m
    y_bottom_tube = wksf_thickness_m + props["tube_height_m"]

    c_wksf = max(abs(y_top_wksf - props["y_ref_wksf"]), abs(y_interface - props["y_ref_wksf"]))
    c_tube = max(abs(y_interface - props["y_ref_tube"]), abs(y_bottom_tube - props["y_ref_tube"]))

    max_stress_wksf_pa = max_moment_nm * c_wksf * E_wksf_pa / EI_total
    max_stress_tube_pa = max_moment_nm * c_tube * E_tube_pa / EI_total

    c1, c2 = st.columns(2)
    c1.metric("Max deflection", f"{max_deflection_m * 1000:.3f} mm")
    c2.metric("Total mass", f"{mass_kg:.3f} kg")

    c1, c2 = st.columns(2)
    c1.metric("Max WKSF stress", f"{max_stress_wksf_pa / 1e6:.2f} MPa")
    c2.metric("Max tube stress", f"{max_stress_tube_pa / 1e6:.2f} MPa")

    with st.expander("Composite section properties"):
        st.write(f"WKSF area: {props['A_wksf'] * 1e6:.1f} mm²")
        st.write(f"Total tube area ({n_tubes} tube{'s' if n_tubes > 1 else ''}): {props['A_tube_total'] * 1e6:.1f} mm²")
        if bonded:
            st.write("Bonding: fully bonded (transformed-section method, shared neutral axis)")
            st.write(f"Modular ratio (n = E_tube / E_wksf): {props['n_ratio']:.3f}")
            st.write(f"Neutral axis depth from top of WKSF: {props['y_ref_wksf'] * 1000:.2f} mm")
            st.write(f"Transformed second moment of area (I, in WKSF-equivalent units): {props['I_transformed'] * 1e12:.1f} mm⁴")
        else:
            st.write("Bonding: unbonded (each layer bends about its own centroid, no shear transfer)")
            st.write(f"WKSF centroid depth from top of WKSF: {props['y_ref_wksf'] * 1000:.2f} mm")
            st.write(f"Tube centroid depth from top of WKSF: {props['y_ref_tube'] * 1000:.2f} mm")
        st.write(f"Effective bending stiffness (EI): {EI_total:.1f} N·m²")
        st.write(f"Self-weight distributed load: {w_self:.3f} N/m")

    deflection_plot(x, y)


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------
st.title("Beam Calculator")

tab1, tab2 = st.tabs(["Simple Tube Bending", "Composite WKSF & Tube Bending"])
with tab1:
    render_simple_tube_tab()
with tab2:
    render_composite_tab()
