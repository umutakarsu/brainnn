#!/usr/bin/env python3
"""
BCI Engineering Drawings — Technical Illustrations
====================================================
A: Invasive BCI Chip (Neuralink/BISC-class)
B: Semi-Invasive Cortical Interface (Precision Layer 7-class)

Professional engineering drawing style with:
- Cross-section views with hatching
- Exploded assembly views
- Dimensioning (ISO 129)
- Material callouts & Bill of Materials
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Arc, FancyArrowPatch
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def dim_line(ax, p1, p2, text, offset=0.3, fontsize=7, color='#333333',
             vertical=False, side='right'):
    """Draw ISO-style dimension line with arrows and text."""
    if vertical:
        # Vertical dimension
        xoff = offset if side == 'right' else -offset
        ax.annotate('', xy=(p1[0]+xoff, p1[1]), xytext=(p1[0]+xoff, p2[1]),
                    arrowprops=dict(arrowstyle='<->', color=color, lw=0.8))
        ax.plot([p1[0], p1[0]+xoff+0.05], [p1[1], p1[1]], color=color, lw=0.4, ls='--')
        ax.plot([p2[0], p2[0]+xoff+0.05], [p2[1], p2[1]], color=color, lw=0.4, ls='--')
        mid_y = (p1[1] + p2[1]) / 2
        ax.text(p1[0]+xoff+0.08, mid_y, text, fontsize=fontsize, color=color,
                ha='left', va='center', rotation=90,
                bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none', alpha=0.9))
    else:
        # Horizontal dimension
        yoff = offset if side == 'right' else -offset
        ax.annotate('', xy=(p1[0], p1[1]+yoff), xytext=(p2[0], p2[1]+yoff),
                    arrowprops=dict(arrowstyle='<->', color=color, lw=0.8))
        ax.plot([p1[0], p1[0]], [p1[1], p1[1]+yoff+0.05], color=color, lw=0.4, ls='--')
        ax.plot([p2[0], p2[0]], [p2[1], p2[1]+yoff+0.05], color=color, lw=0.4, ls='--')
        mid_x = (p1[0] + p2[0]) / 2
        ax.text(mid_x, p1[1]+yoff+0.08, text, fontsize=fontsize, color=color,
                ha='center', va='bottom',
                bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none', alpha=0.9))


def callout(ax, xy, text, xytext, fontsize=6.5, color='#1a1a1a'):
    """Draw a callout/leader line with label."""
    ax.annotate(text, xy=xy, xytext=xytext,
                fontsize=fontsize, color=color,
                ha='left', va='center',
                arrowprops=dict(arrowstyle='->', color='#555555', lw=0.7,
                               connectionstyle='arc3,rad=0.1'),
                bbox=dict(boxstyle='round,pad=0.2', fc='#f8f8f8', ec='#cccccc',
                         lw=0.5, alpha=0.95))


def hatch_rect(ax, xy, width, height, angle=45, density=15, color='#aaaaaa', lw=0.3):
    """Draw cross-hatch pattern inside a rectangle (section view convention)."""
    x0, y0 = xy
    for i in np.linspace(-max(width,height), max(width,height)*2, density):
        if angle == 45:
            x_start = x0 + i
            y_start = y0
            x_end = x0 + i + height
            y_end = y0 + height
        else:
            x_start = x0 + i + height
            y_start = y0
            x_end = x0 + i
            y_end = y0 + height

        # Clip to rectangle
        xs, ys, xe, ye = x_start, y_start, x_end, y_end
        # Simple clipping
        if xe < x0 or xs > x0 + width:
            continue
        xs = max(xs, x0)
        xe = min(xe, x0 + width)
        # Adjust y accordingly
        if angle == 45:
            ys = y0 + (xs - x_start)
            ye = y0 + (xe - x_start)
        else:
            ys = y0 + (x_start + height - xs)
            ye = y0 + (x_start + height - xe)
        ys = max(min(ys, y0+height), y0)
        ye = max(min(ye, y0+height), y0)
        ax.plot([xs, xe], [ys, ye], color=color, lw=lw)


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING A: INVASIVE BCI CHIP — Cross-Section View
# ══════════════════════════════════════════════════════════════════════════════

def draw_invasive_bci_cross_section():
    """
    Cross-section of an invasive BCI implant showing:
    - Titanium hermetic enclosure
    - CMOS ASIC die
    - Wireless coil (inductive power + data)
    - Flexible electrode threads
    - Feedthrough assembly
    - Battery (optional)
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-4, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title block
    ax.text(7, 9.5, 'INVASIVE BCI IMPLANT — Cross-Section View (A-A)',
            fontsize=14, fontweight='bold', ha='center', va='center', color='#1a1a1a')
    ax.text(7, 9.0, 'Intracortical Closed-Loop Neural Interface | Scale 5:1',
            fontsize=9, ha='center', va='center', color='#666666')

    # ── Skull layer (top) ──
    skull_y = 7.0
    skull = FancyBboxPatch((0.5, skull_y), 13, 0.8,
                           boxstyle="round,pad=0.05", fc='#F5E6D3', ec='#8B7355', lw=1.5)
    ax.add_patch(skull)
    hatch_rect(ax, (0.5, skull_y), 13, 0.8, angle=45, density=25, color='#C4A882', lw=0.4)
    ax.text(14, skull_y+0.4, 'Kranial Kemik\n(Cranial Bone)', fontsize=7,
            va='center', color='#8B7355', style='italic')

    # Skull cavity (drilled pocket for implant)
    cavity_x = 3.0
    cavity_w = 8.0
    ax.add_patch(plt.Rectangle((cavity_x, skull_y), cavity_w, 0.8, fc='white', ec='#8B7355', lw=1))

    # ── Titanium Enclosure ──
    ti_x = 3.2
    ti_y = 4.8
    ti_w = 7.6
    ti_h = 2.8
    ti_wall = 0.25

    # Outer shell
    ti_outer = FancyBboxPatch((ti_x, ti_y), ti_w, ti_h,
                              boxstyle="round,pad=0.1", fc='#D4D4D8', ec='#52525B', lw=2)
    ax.add_patch(ti_outer)

    # Inner cavity
    inner_x = ti_x + ti_wall
    inner_y = ti_y + ti_wall
    inner_w = ti_w - 2*ti_wall
    inner_h = ti_h - 2*ti_wall
    ax.add_patch(plt.Rectangle((inner_x, inner_y), inner_w, inner_h,
                               fc='#FAFAFA', ec='#71717A', lw=0.8))

    # Hatching on Ti walls (cross-section convention)
    # Left wall
    hatch_rect(ax, (ti_x, ti_y), ti_wall, ti_h, angle=45, density=8, color='#71717A', lw=0.5)
    # Right wall
    hatch_rect(ax, (ti_x+ti_w-ti_wall, ti_y), ti_wall, ti_h, angle=45, density=8, color='#71717A', lw=0.5)
    # Top wall
    hatch_rect(ax, (ti_x, ti_y+ti_h-ti_wall), ti_w, ti_wall, angle=45, density=20, color='#71717A', lw=0.5)
    # Bottom wall
    hatch_rect(ax, (ti_x, ti_y), ti_w, ti_wall, angle=45, density=20, color='#71717A', lw=0.5)

    # ── CMOS ASIC Die ──
    asic_x = 4.0
    asic_y = 5.3
    asic_w = 2.8
    asic_h = 0.6
    ax.add_patch(plt.Rectangle((asic_x, asic_y), asic_w, asic_h,
                               fc='#1E3A5F', ec='#0F172A', lw=1.2))
    ax.text(asic_x + asic_w/2, asic_y + asic_h/2, 'CMOS ASIC\n3,072 ch',
            fontsize=6, color='white', ha='center', va='center', fontweight='bold')

    # Wire bonds from ASIC
    for i in range(8):
        bx = asic_x + 0.2 + i * 0.35
        ax.plot([bx, bx+0.1, bx+0.2], [asic_y, asic_y-0.15, asic_y-0.2],
                color='#FFD700', lw=0.6)

    # ── Substrate / PCB ──
    pcb_y = 5.1
    ax.add_patch(plt.Rectangle((3.6, pcb_y), 6.8, 0.2,
                               fc='#166534', ec='#14532D', lw=0.8))
    ax.text(8.5, pcb_y + 0.1, 'FR4 / Polyimide PCB', fontsize=5.5,
            ha='center', va='center', color='#BBF7D0')

    # ── Wireless Coil (Inductive) ──
    coil_cx = 8.5
    coil_cy = 6.5
    coil_r = 1.2
    for i in range(4):
        r = coil_r - i * 0.12
        circle = plt.Circle((coil_cx, coil_cy), r, fill=False,
                            ec='#B45309', lw=1.2 - i*0.2, ls='-')
        ax.add_patch(circle)
    ax.text(coil_cx, coil_cy, 'RF Coil\n(915 MHz)', fontsize=5.5,
            ha='center', va='center', color='#92400E', fontweight='bold')

    # ── Li-ion Micro Battery ──
    batt_x = 4.0
    batt_y = 6.2
    batt_w = 2.0
    batt_h = 0.9
    ax.add_patch(FancyBboxPatch((batt_x, batt_y), batt_w, batt_h,
                                boxstyle="round,pad=0.05", fc='#7C3AED', ec='#5B21B6', lw=1))
    ax.text(batt_x + batt_w/2, batt_y + batt_h/2, 'Li-ion µBattery\n3.7V / 12mAh',
            fontsize=5.5, color='white', ha='center', va='center', fontweight='bold')
    # Battery terminals
    ax.plot([batt_x+batt_w, batt_x+batt_w+0.3], [batt_y+0.6, batt_y+0.6],
            color='#EF4444', lw=1.5)
    ax.text(batt_x+batt_w+0.35, batt_y+0.6, '+', fontsize=7, color='#EF4444', fontweight='bold')
    ax.plot([batt_x+batt_w, batt_x+batt_w+0.3], [batt_y+0.3, batt_y+0.3],
            color='#3B82F6', lw=1.5)
    ax.text(batt_x+batt_w+0.35, batt_y+0.3, '−', fontsize=8, color='#3B82F6', fontweight='bold')

    # ── Hermetic Feedthrough (bottom of Ti case → threads) ──
    ft_cx = 7.0
    ft_y = ti_y
    ft_w = 2.0
    ft_h = 0.25
    ax.add_patch(plt.Rectangle((ft_cx - ft_w/2, ft_y - ft_h), ft_w, ft_h*2,
                               fc='#FDE68A', ec='#92400E', lw=1))
    ax.text(ft_cx, ft_y, 'Ceramic\nFeedthrough', fontsize=5, ha='center', va='center',
            color='#78350F')

    # ── Flexible Electrode Threads ──
    thread_colors = ['#EF4444', '#F97316', '#EAB308', '#22C55E',
                     '#06B6D4', '#3B82F6', '#8B5CF6', '#EC4899']
    n_threads = 8
    thread_start_x = ft_cx - 1.2
    thread_spacing = 0.35

    for i in range(n_threads):
        tx = thread_start_x + i * thread_spacing
        # Thread path (slightly curved)
        t_points_y = np.linspace(ft_y - ft_h, ti_y - 2.5, 30)
        t_points_x = tx + 0.08 * np.sin(np.linspace(0, np.pi*1.5, 30)) * (i % 3 - 1)
        ax.plot(t_points_x, t_points_y, color=thread_colors[i], lw=0.8, alpha=0.8)

        # Electrode tips (small dots)
        for j in range(4):
            ey = t_points_y[-1] + j * 0.15
            ax.plot(t_points_x[-1], ey - 0.5, 'o', color=thread_colors[i],
                    markersize=1.5, alpha=0.9)

    # Thread label
    ax.text(ft_cx, ti_y - 1.5, '128 Ultra-Thin Threads\n(4-6 µm width, polyimide)',
            fontsize=6, ha='center', va='center', color='#374151',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#D1D5DB', lw=0.5))

    # ── Electrode tips in cortex ──
    cortex_y = 1.5
    ax.add_patch(plt.Rectangle((2.0, cortex_y), 10, 1.2,
                               fc='#FECDD3', ec='#BE123C', lw=0.8, alpha=0.5))
    ax.text(12.5, cortex_y + 0.6, 'Kortikal Doku\n(Cortical Tissue)', fontsize=7,
            va='center', color='#BE123C', style='italic')
    # Neuron dots in cortex
    np.random.seed(42)
    for _ in range(40):
        nx = np.random.uniform(2.5, 11.5)
        ny = np.random.uniform(cortex_y + 0.1, cortex_y + 1.1)
        ax.plot(nx, ny, 'o', color='#9F1239', markersize=1.5, alpha=0.3)

    # ── Dura Mater ──
    dura_y = 2.8
    ax.plot([2.0, 12.0], [dura_y, dura_y], color='#7C3AED', lw=1.5, ls='-')
    ax.text(12.5, dura_y, 'Dura Mater', fontsize=7, va='center', color='#7C3AED', style='italic')

    # ── Scalp (top layer) ──
    scalp_y = skull_y + 0.8
    ax.add_patch(plt.Rectangle((0.5, scalp_y), 13, 0.4,
                               fc='#FBBF24', ec='#D97706', lw=1, alpha=0.6))
    ax.text(14, scalp_y + 0.2, 'Skalp (Scalp)', fontsize=7, va='center',
            color='#D97706', style='italic')

    # ── Dimension Lines ──
    # Overall width
    dim_line(ax, (ti_x, ti_y+ti_h), (ti_x+ti_w, ti_y+ti_h),
             '23.0 mm', offset=0.5, fontsize=7)
    # Overall height
    dim_line(ax, (ti_x+ti_w, ti_y), (ti_x+ti_w, ti_y+ti_h),
             '2.0 mm', offset=0.5, fontsize=7, vertical=True)
    # Ti wall thickness
    dim_line(ax, (ti_x, ti_y+ti_h), (ti_x+ti_wall, ti_y+ti_h),
             '0.25', offset=0.25, fontsize=5.5)
    # Thread length
    dim_line(ax, (ft_cx+2.5, ft_y-ft_h), (ft_cx+2.5, cortex_y+1.2),
             '3.0 mm\n(thread length)', offset=0.3, fontsize=6, vertical=True)

    # ── Callouts ──
    callout(ax, (ti_x+0.12, ti_y+ti_h/2), 'Ti Grade 5 (Ti-6Al-4V)\nHermetic Laser Weld\nWall: 250 µm',
            (-0.8, ti_y+ti_h/2+1.5), fontsize=6)
    callout(ax, (asic_x+asic_w/2, asic_y+asic_h), 'CMOS 65nm ASIC\n3,072 channels\n6.6 µW/channel\n20 kSps ADC',
            (0.5, 8.5), fontsize=6)
    callout(ax, (coil_cx+coil_r, coil_cy), 'Inductive Power + UWB Data\n100 Mbps wireless\n915 MHz carrier',
            (12, 8.5), fontsize=6)
    callout(ax, (ft_cx, ft_y-ft_h), 'Al₂O₃ Ceramic Feedthrough\n128-pin hermetic\nBrazed Ti-ceramic joint',
            (0.5, 3.5), fontsize=6)

    # ── Section line indicator ──
    ax.annotate('A', xy=(0.3, 5), fontsize=12, fontweight='bold', color='#DC2626',
                bbox=dict(boxstyle='circle,pad=0.3', fc='white', ec='#DC2626', lw=2))
    ax.annotate('A', xy=(14.2, 5), fontsize=12, fontweight='bold', color='#DC2626',
                bbox=dict(boxstyle='circle,pad=0.3', fc='white', ec='#DC2626', lw=2))
    ax.plot([0.7, 1.5], [5, 5], color='#DC2626', lw=1.5, ls='-.')
    ax.plot([13.0, 13.8], [5, 5], color='#DC2626', lw=1.5, ls='-.')

    # ── Bill of Materials ──
    bom_y = -0.5
    bom_items = [
        ('①', 'Ti-6Al-4V Enclosure', '23×18×2 mm', 'Hermetic, laser-welded'),
        ('②', 'CMOS ASIC (65nm)', '4.5×4.5×0.3 mm', '3,072 ch, 200 Mbps'),
        ('③', 'RF Inductive Coil', 'Ø 12 mm, 4-turn', '915 MHz, 100 Mbps UWB'),
        ('④', 'Li-ion µBattery', '8×5×1.2 mm', '3.7V, 12 mAh'),
        ('⑤', 'Ceramic Feedthrough', '8×1.5 mm', 'Al₂O₃, 128-pin'),
        ('⑥', 'Polyimide Threads', '4-6 µm × 3 mm', '128 threads, 1,024 electrodes'),
    ]

    ax.text(0.5, bom_y, 'BILL OF MATERIALS', fontsize=8, fontweight='bold', color='#1a1a1a')
    for i, (num, name, dim, note) in enumerate(bom_items):
        y = bom_y - 0.4 - i * 0.35
        ax.text(0.5, y, num, fontsize=7, fontweight='bold', color='#DC2626')
        ax.text(1.1, y, name, fontsize=6.5, color='#1a1a1a')
        ax.text(5.5, y, dim, fontsize=6.5, color='#555555')
        ax.text(9.0, y, note, fontsize=6, color='#777777', style='italic')

    # Drawing border
    ax.add_patch(plt.Rectangle((-0.5, -3.5), 15.5, 13.5,
                               fill=False, ec='#333333', lw=2))

    # Title block (bottom right)
    ax.add_patch(plt.Rectangle((9.5, -3.5), 5.5, 1.2, fc='#F9FAFB', ec='#333333', lw=1))
    ax.text(12.25, -2.6, 'DWG: BCI-INV-001-A', fontsize=7, ha='center', fontweight='bold')
    ax.text(12.25, -2.95, 'Invasive BCI Implant | Rev. 1.0', fontsize=6, ha='center', color='#555')
    ax.text(12.25, -3.25, 'Scale 5:1 | Units: mm | Material: Ti Gr.5', fontsize=5.5,
            ha='center', color='#777')

    plt.tight_layout()
    plt.savefig('BCI_A_invasive_cross_section.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[DWG-A] Invasive BCI cross-section → BCI_A_invasive_cross_section.png")


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING A2: INVASIVE BCI — Exploded Assembly View
# ══════════════════════════════════════════════════════════════════════════════

def draw_invasive_bci_exploded():
    """Exploded isometric-style assembly view."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    ax.set_xlim(-1, 14)
    ax.set_ylim(-2, 14)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(7, 13.5, 'INVASIVE BCI IMPLANT — Exploded Assembly View',
            fontsize=14, fontweight='bold', ha='center', color='#1a1a1a')
    ax.text(7, 13.0, 'Component Stack-Up | Not to Scale',
            fontsize=9, ha='center', color='#666666')

    # Assembly levels (bottom to top with spacing)
    levels = [
        # (y_pos, label, color, edge_color, width, height, detail_text)
        (1.0, '⑥ Electrode Thread Array', '#FECDD3', '#BE123C', 6, 0.5,
         '128 polyimide threads | 1,024 Pt/Ir electrodes\n4-6 µm width | 3 mm penetration depth'),
        (3.0, '⑤ Ceramic Feedthrough Plate', '#FDE68A', '#92400E', 7, 0.6,
         'Al₂O₃ (99.6%) ceramic | 128-pin hermetic\nBrazed Ti-ceramic seal | Leak rate < 1×10⁻⁹ atm·cc/s'),
        (5.0, '③ PCB + ② ASIC Assembly', '#DCFCE7', '#166534', 8, 0.8,
         'Polyimide flex PCB (25 µm) | CMOS 65nm ASIC\n3,072 channels | ADC: 10-bit, 20 kSps | Wire bonded'),
        (7.0, '④ Li-ion Micro Battery', '#EDE9FE', '#7C3AED', 4, 0.6,
         'Solid-state Li-ion | 3.7V, 12 mAh\n8×5×1.2 mm | 500+ cycle life'),
        (9.0, '③ RF Coil + Antenna', '#FFF7ED', '#C2410C', 6, 0.5,
         'Inductive power Rx coil (Ø12 mm, 4-turn Cu)\nUWB antenna (915 MHz) | 100 Mbps data rate'),
        (11.0, '① Titanium Enclosure (Lid)', '#F4F4F5', '#52525B', 8.5, 0.7,
         'Ti-6Al-4V (Grade 5) | 250 µm wall\nLaser-welded hermetic seal | Biocompatible (ISO 10993)'),
    ]

    center_x = 6.5
    for y, label, fc, ec, w, h, detail in levels:
        x = center_x - w/2

        # Component box
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                            fc=fc, ec=ec, lw=1.5, alpha=0.9)
        ax.add_patch(box)

        # Label inside
        ax.text(center_x, y + h/2, label, fontsize=8, fontweight='bold',
                ha='center', va='center', color=ec)

        # Detail text (right side)
        ax.text(center_x + w/2 + 0.5, y + h/2, detail,
                fontsize=6, va='center', color='#555555',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#E5E7EB', lw=0.5))

        # Assembly arrow (dashed center line)
        if y < 11.0:
            ax.annotate('', xy=(center_x, y + h + 0.1),
                       xytext=(center_x, y + h + 0.85),
                       arrowprops=dict(arrowstyle='->', color='#9CA3AF',
                                      lw=1, ls='--'))

    # Bottom annotation: Titanium base (mirror of lid)
    base_y = -0.5
    base_box = FancyBboxPatch((center_x - 8.5/2, base_y), 8.5, 0.7,
                              boxstyle="round,pad=0.08",
                              fc='#F4F4F5', ec='#52525B', lw=1.5, alpha=0.9)
    ax.add_patch(base_box)
    ax.text(center_x, base_y + 0.35, '① Titanium Enclosure (Base)',
            fontsize=8, fontweight='bold', ha='center', va='center', color='#52525B')
    ax.annotate('', xy=(center_x, base_y + 0.8),
               xytext=(center_x, base_y + 1.7),
               arrowprops=dict(arrowstyle='->', color='#9CA3AF', lw=1, ls='--'))

    # Assembly order note
    ax.text(0, -1.5, 'ASSEMBLY ORDER: Base → Feedthrough → PCB+ASIC → Battery → RF Coil → Lid (laser weld)',
            fontsize=7, color='#374151', style='italic')

    # Drawing border
    ax.add_patch(plt.Rectangle((-0.5, -2), 14.5, 16,
                               fill=False, ec='#333333', lw=2))

    # Title block
    ax.add_patch(plt.Rectangle((9, -2), 5, 1, fc='#F9FAFB', ec='#333333', lw=1))
    ax.text(11.5, -1.3, 'DWG: BCI-INV-002-A', fontsize=7, ha='center', fontweight='bold')
    ax.text(11.5, -1.65, 'Exploded Assembly | Rev. 1.0', fontsize=6, ha='center', color='#555')

    plt.tight_layout()
    plt.savefig('BCI_A_invasive_exploded.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[DWG-A2] Invasive BCI exploded view → BCI_A_invasive_exploded.png")


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING B: SEMI-INVASIVE CORTICAL INTERFACE — Cross-Section
# ══════════════════════════════════════════════════════════════════════════════

def draw_semi_invasive_cross_section():
    """
    Cross-section of a Precision Layer 7-class semi-invasive BCI:
    - Ultra-thin polyimide electrode array (on cortical surface)
    - Ribbon cable through burr hole
    - External pedestal connector
    - External processing unit
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-4, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(8, 9.5, 'SEMI-INVASIVE CORTICAL INTERFACE — Cross-Section View (B-B)',
            fontsize=14, fontweight='bold', ha='center', color='#1a1a1a')
    ax.text(8, 9.0, 'Subdural µECoG Array with External Processor | Scale 3:1',
            fontsize=9, ha='center', color='#666666')

    # ── Skull ──
    skull_y = 6.0
    skull_h = 1.0
    ax.add_patch(plt.Rectangle((0.5, skull_y), 15, skull_h,
                               fc='#F5E6D3', ec='#8B7355', lw=1.5))
    hatch_rect(ax, (0.5, skull_y), 15, skull_h, angle=45, density=30, color='#C4A882', lw=0.4)

    # Burr hole
    burr_x = 7.5
    burr_w = 1.0
    ax.add_patch(plt.Rectangle((burr_x, skull_y), burr_w, skull_h,
                               fc='white', ec='#8B7355', lw=1))

    # ── Scalp ──
    scalp_y = skull_y + skull_h
    ax.add_patch(plt.Rectangle((0.5, scalp_y), 15, 0.35,
                               fc='#FBBF24', ec='#D97706', lw=0.8, alpha=0.6))

    # ── Dura Mater ──
    dura_y = 5.6
    ax.plot([0.5, 15.5], [dura_y, dura_y], color='#7C3AED', lw=1.5)
    ax.text(16, dura_y, 'Dura', fontsize=7, color='#7C3AED', style='italic')

    # ── Cortical Surface ──
    cortex_y = 3.5
    cortex_h = 2.0
    # Wavy cortex surface (gyri/sulci)
    cx = np.linspace(0.5, 15.5, 200)
    cy_top = dura_y - 0.3 + 0.15 * np.sin(cx * 2.5) + 0.08 * np.sin(cx * 5)
    cy_bot = cy_top - cortex_h
    ax.fill_between(cx, cy_bot, cy_top, color='#FECDD3', alpha=0.5)
    ax.plot(cx, cy_top, color='#BE123C', lw=0.8)

    # ── Thin-Film Electrode Array (on cortex surface) ──
    array_x = 2.5
    array_w = 11.0
    array_y_center = np.mean(cy_top[30:170])

    # The array follows the cortical surface curvature
    ax_arr = cx[(cx >= array_x) & (cx <= array_x + array_w)]
    ay_arr = cy_top[(cx >= array_x) & (cx <= array_x + array_w)]

    # Array outline (ultra thin — just a line on the surface)
    ax.plot(ax_arr, ay_arr, color='#2563EB', lw=3, alpha=0.8)
    ax.plot(ax_arr, ay_arr - 0.03, color='#1E40AF', lw=1, alpha=0.5)

    # Electrode contacts (small dots along the array)
    n_electrodes = 32
    e_indices = np.linspace(5, len(ax_arr)-5, n_electrodes).astype(int)
    for idx in e_indices:
        ax.plot(ax_arr[idx], ay_arr[idx] + 0.02, 's', color='#FFD700',
                markersize=2.5, markeredgecolor='#B45309', markeredgewidth=0.3)

    # Array label
    ax.text(8, array_y_center - 0.8,
            'Polyimide Thin-Film µECoG Array\n1,024 electrodes | 10 µm thickness\nPt contacts, 400 µm pitch',
            fontsize=6.5, ha='center', va='center', color='#1E40AF',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#93C5FD', lw=0.8, alpha=0.9))

    # ── Ribbon Cable (through burr hole) ──
    ribbon_x = burr_x + burr_w/2
    # Cable from array up through burr hole
    cable_path_x = [8.0, 8.0, ribbon_x, ribbon_x]
    cable_path_y = [array_y_center + 0.3, dura_y - 0.1, skull_y, scalp_y + 0.35]
    ax.plot(cable_path_x, cable_path_y, color='#F59E0B', lw=3, alpha=0.7)
    ax.plot(cable_path_x, cable_path_y, color='#D97706', lw=1, alpha=0.5)
    ax.text(ribbon_x + 0.8, (skull_y + dura_y)/2, 'Polyimide\nRibbon Cable\n(256 traces)',
            fontsize=5.5, color='#92400E', va='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='#FFF7ED', ec='#FDBA74', lw=0.5))

    # ── External Pedestal Connector ──
    ped_x = ribbon_x - 0.8
    ped_y = scalp_y + 0.35
    ped_w = 1.6
    ped_h = 0.8

    ax.add_patch(FancyBboxPatch((ped_x, ped_y), ped_w, ped_h,
                                boxstyle="round,pad=0.05", fc='#D4D4D8', ec='#52525B', lw=1.5))
    hatch_rect(ax, (ped_x, ped_y), ped_w, ped_h, angle=45, density=8, color='#71717A', lw=0.4)
    ax.text(ped_x + ped_w/2, ped_y + ped_h/2, 'Ti Pedestal', fontsize=6,
            ha='center', va='center', color='#27272A', fontweight='bold')

    # ── External Cable to Processing Unit ──
    ext_cable_y = ped_y + ped_h
    ax.plot([ped_x + ped_w/2, ped_x + ped_w/2, 13],
            [ext_cable_y, ext_cable_y + 0.5, ext_cable_y + 0.5],
            color='#6B7280', lw=2, ls='-')

    # ── External Processing Unit ──
    ext_x = 12.5
    ext_y = ext_cable_y - 0.5
    ext_w = 3.0
    ext_h = 2.0

    ax.add_patch(FancyBboxPatch((ext_x, ext_y), ext_w, ext_h,
                                boxstyle="round,pad=0.15", fc='#F0FDF4', ec='#166534', lw=2))

    # Internal components of external unit
    # DSP chip
    ax.add_patch(plt.Rectangle((ext_x+0.3, ext_y+1.1), 1.0, 0.5,
                               fc='#1E3A5F', ec='#0F172A', lw=0.8))
    ax.text(ext_x+0.8, ext_y+1.35, 'DSP/\nFPGA', fontsize=5, color='white',
            ha='center', va='center')

    # Wireless module
    ax.add_patch(plt.Rectangle((ext_x+1.5, ext_y+1.1), 1.0, 0.5,
                               fc='#DBEAFE', ec='#2563EB', lw=0.8))
    ax.text(ext_x+2.0, ext_y+1.35, 'BLE\n5.0', fontsize=5, color='#1E40AF',
            ha='center', va='center')

    # Battery
    ax.add_patch(plt.Rectangle((ext_x+0.3, ext_y+0.3), 2.2, 0.5,
                               fc='#EDE9FE', ec='#7C3AED', lw=0.8))
    ax.text(ext_x+1.4, ext_y+0.55, 'Li-Po Battery (250 mAh)',
            fontsize=5, color='#5B21B6', ha='center', va='center')

    ax.text(ext_x + ext_w/2, ext_y + ext_h + 0.15,
            'External Processing Unit\n(Behind-the-Ear)', fontsize=7,
            ha='center', va='bottom', fontweight='bold', color='#166534')

    # ── Dimension Lines ──
    # Array length
    dim_line(ax, (array_x, cortex_y - 0.5), (array_x + array_w, cortex_y - 0.5),
             f'{array_w*2:.0f} mm (array length)', offset=-0.5, fontsize=7, side='left')

    # Array thickness
    callout(ax, (5, array_y_center), 'Array thickness: 10 µm\n(1/5 human hair)',
            (0.5, 2.5), fontsize=6.5)

    # Skull thickness
    dim_line(ax, (15.8, skull_y), (15.8, skull_y+skull_h),
             '7 mm\n(skull)', offset=0.3, fontsize=6, vertical=True)

    # ── Callouts ──
    callout(ax, (3, cy_top[40]), 'Subdural placement\n(below dura, on cortex)\nNo tissue penetration',
            (0, 1.0), fontsize=6)
    callout(ax, (ext_x + ext_w, ext_y + ext_h/2),
            'Wireless data to PC/tablet\nBLE 5.0 + USB backup\nReal-time 1,024 ch streaming',
            (ext_x + ext_w + 0.2, ext_y + ext_h + 1.5), fontsize=6)

    # ── Bill of Materials ──
    bom_y = -0.8
    bom_items = [
        ('①', 'Polyimide µECoG Array', '22×22×0.01 mm', '1,024 Pt electrodes, 400 µm pitch'),
        ('②', 'Polyimide Ribbon Cable', '0.1×50 mm', '256 traces, flexible, biocompatible'),
        ('③', 'Ti Pedestal Connector', 'Ø 15 mm', 'Ti Grade 2, hermetic feedthrough'),
        ('④', 'External Processing Unit', '35×25×8 mm', 'FPGA + BLE 5.0 + Li-Po 250 mAh'),
    ]

    ax.text(0.5, bom_y, 'BILL OF MATERIALS', fontsize=8, fontweight='bold', color='#1a1a1a')
    for i, (num, name, dim, note) in enumerate(bom_items):
        y = bom_y - 0.4 - i * 0.35
        ax.text(0.5, y, num, fontsize=7, fontweight='bold', color='#2563EB')
        ax.text(1.1, y, name, fontsize=6.5, color='#1a1a1a')
        ax.text(6.0, y, dim, fontsize=6.5, color='#555555')
        ax.text(10.0, y, note, fontsize=6, color='#777777', style='italic')

    # Section line
    ax.annotate('B', xy=(0.0, 5.5), fontsize=12, fontweight='bold', color='#2563EB',
                bbox=dict(boxstyle='circle,pad=0.3', fc='white', ec='#2563EB', lw=2))
    ax.annotate('B', xy=(16.2, 5.5), fontsize=12, fontweight='bold', color='#2563EB',
                bbox=dict(boxstyle='circle,pad=0.3', fc='white', ec='#2563EB', lw=2))

    # Drawing border
    ax.add_patch(plt.Rectangle((-0.5, -3.5), 17, 13.5,
                               fill=False, ec='#333333', lw=2))

    # Title block
    ax.add_patch(plt.Rectangle((11, -3.5), 5.5, 1.2, fc='#F9FAFB', ec='#333333', lw=1))
    ax.text(13.75, -2.6, 'DWG: BCI-SEMI-001-B', fontsize=7, ha='center', fontweight='bold')
    ax.text(13.75, -2.95, 'Semi-Invasive Cortical Interface | Rev. 1.0', fontsize=6,
            ha='center', color='#555')
    ax.text(13.75, -3.25, 'Scale 3:1 | Units: mm | Material: Polyimide/Ti', fontsize=5.5,
            ha='center', color='#777')

    plt.tight_layout()
    plt.savefig('BCI_B_semi_invasive_cross_section.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[DWG-B] Semi-invasive cortical interface → BCI_B_semi_invasive_cross_section.png")


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING B2: ELECTRODE ARRAY DETAIL VIEW
# ══════════════════════════════════════════════════════════════════════════════

def draw_electrode_array_detail():
    """Zoomed detail view of the thin-film electrode array."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(-1, 14)
    ax.set_ylim(-2, 11)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(7, 10.5, 'THIN-FILM µECoG ELECTRODE ARRAY — Detail View',
            fontsize=14, fontweight='bold', ha='center', color='#1a1a1a')
    ax.text(7, 10.0, 'Polyimide Substrate with Platinum Electrode Contacts | Scale 50:1',
            fontsize=9, ha='center', color='#666666')

    # ── Array substrate (polyimide) ──
    arr_x = 1.0
    arr_y = 2.0
    arr_w = 11.0
    arr_h = 6.5

    # Main substrate
    ax.add_patch(FancyBboxPatch((arr_x, arr_y), arr_w, arr_h,
                                boxstyle="round,pad=0.15",
                                fc='#FFF7ED', ec='#C2410C', lw=1.5, alpha=0.9))

    # Grid of electrodes
    n_rows = 8
    n_cols = 16
    e_spacing_x = arr_w / (n_cols + 1)
    e_spacing_y = arr_h / (n_rows + 1)
    electrode_r = 0.15

    for row in range(n_rows):
        for col in range(n_cols):
            ex = arr_x + (col + 1) * e_spacing_x
            ey = arr_y + (row + 1) * e_spacing_y

            # Electrode pad
            circle = plt.Circle((ex, ey), electrode_r, fc='#FFD700',
                               ec='#B45309', lw=0.5)
            ax.add_patch(circle)

            # Trace line (going right)
            if col < n_cols - 1:
                ax.plot([ex + electrode_r, ex + e_spacing_x - electrode_r],
                        [ey, ey], color='#D4A017', lw=0.3, alpha=0.5)

    # ── Zoomed inset: single electrode ──
    inset_x = 1.5
    inset_y = 0.0
    inset_w = 4.0
    inset_h = 1.8

    ax.add_patch(plt.Rectangle((inset_x, inset_y), inset_w, inset_h,
                               fc='#FAFAFA', ec='#333333', lw=1.5))
    ax.text(inset_x + inset_w/2, inset_y + inset_h + 0.15,
            'DETAIL C — Single Electrode (Scale 500:1)', fontsize=7,
            ha='center', fontweight='bold', color='#333333')

    # Large electrode detail
    big_e_cx = inset_x + 1.2
    big_e_cy = inset_y + inset_h/2
    big_r = 0.5

    # Polyimide base
    ax.add_patch(plt.Rectangle((inset_x + 0.2, big_e_cy - 0.3), 3.6, 0.6,
                               fc='#FFEDD5', ec='#C2410C', lw=0.8))
    ax.text(inset_x + 2.0, big_e_cy - 0.15, 'Polyimide (10 µm)', fontsize=5,
            ha='center', color='#9A3412')

    # Pt electrode pad
    circle_big = plt.Circle((big_e_cx, big_e_cy + 0.05), big_r,
                            fc='#FFD700', ec='#92400E', lw=1.5)
    ax.add_patch(circle_big)
    ax.text(big_e_cx, big_e_cy + 0.05, 'Pt', fontsize=8, ha='center',
            va='center', fontweight='bold', color='#78350F')

    # Trace
    ax.add_patch(plt.Rectangle((big_e_cx + big_r, big_e_cy - 0.05), 1.5, 0.1,
                               fc='#D4A017', ec='#92400E', lw=0.5))
    ax.text(big_e_cx + big_r + 0.75, big_e_cy + 0.2, 'Au trace (5 µm)',
            fontsize=5, ha='center', color='#92400E')

    # Dimensions on inset
    dim_line(ax, (big_e_cx - big_r, inset_y + 0.15), (big_e_cx + big_r, inset_y + 0.15),
             'Ø 200 µm', offset=-0.15, fontsize=6, side='left')

    # ── Layer stack-up diagram ──
    stack_x = 7.0
    stack_y = 0.0
    stack_w = 5.5
    stack_h = 1.8

    ax.add_patch(plt.Rectangle((stack_x, stack_y), stack_w, stack_h,
                               fc='#FAFAFA', ec='#333333', lw=1.5))
    ax.text(stack_x + stack_w/2, stack_y + stack_h + 0.15,
            'LAYER STACK-UP (Cross-Section at Electrode)',
            fontsize=7, ha='center', fontweight='bold', color='#333333')

    layers = [
        ('Tissue Interface', 0.15, '#FECDD3', '#BE123C'),
        ('Pt Electrode (0.2 µm)', 0.12, '#FFD700', '#92400E'),
        ('Ti Adhesion Layer (20 nm)', 0.06, '#D4D4D8', '#52525B'),
        ('Polyimide Top (5 µm)', 0.25, '#FFEDD5', '#C2410C'),
        ('Au Trace (0.5 µm)', 0.08, '#FDE68A', '#B45309'),
        ('Polyimide Base (5 µm)', 0.25, '#FFEDD5', '#C2410C'),
        ('Cortical Surface', 0.15, '#FECDD3', '#BE123C'),
    ]

    layer_y = stack_y + 0.15
    for name, thickness, fc, ec in layers:
        h = thickness * 4  # Scaled for visibility
        ax.add_patch(plt.Rectangle((stack_x + 0.3, layer_y), 2.5, h,
                                   fc=fc, ec=ec, lw=0.8))
        ax.text(stack_x + 3.0, layer_y + h/2, name, fontsize=5.5,
                va='center', color='#374151')
        layer_y += h + 0.02

    # ── Array-level dimensions ──
    dim_line(ax, (arr_x, arr_y), (arr_x + arr_w, arr_y),
             '22.0 mm', offset=-0.5, fontsize=7, side='left')
    dim_line(ax, (arr_x + arr_w, arr_y), (arr_x + arr_w, arr_y + arr_h),
             '22.0 mm', offset=0.5, fontsize=7, vertical=True)

    # Pitch dimension
    e1x = arr_x + 1 * e_spacing_x
    e2x = arr_x + 2 * e_spacing_x
    ey_dim = arr_y + arr_h - e_spacing_y
    dim_line(ax, (e1x, ey_dim), (e2x, ey_dim),
             '400 µm\npitch', offset=0.5, fontsize=5.5)

    # ── Callouts ──
    callout(ax, (arr_x + 3*e_spacing_x, arr_y + 4*e_spacing_y),
            'Electrode: Pt (99.95%)\nØ 50-380 µm (variable)\nImpedance: 1.2-2.0 MΩ @ 1kHz',
            (arr_x - 0.5, arr_y + arr_h + 0.5), fontsize=6)

    callout(ax, (arr_x + arr_w - 0.5, arr_y + arr_h/2),
            'Total: 1,024 electrodes\n32 × 32 grid (reconfigurable)\nStimulation: 10-500 µA',
            (arr_x + arr_w + 0.5, arr_y + 1), fontsize=6)

    # Connector tab (top of array)
    tab_w = 2.0
    tab_h = 0.6
    tab_x = arr_x + arr_w/2 - tab_w/2
    tab_y = arr_y + arr_h
    ax.add_patch(plt.Rectangle((tab_x, tab_y), tab_w, tab_h,
                               fc='#FFEDD5', ec='#C2410C', lw=1))
    ax.text(tab_x + tab_w/2, tab_y + tab_h/2, 'ZIF Connector Tab\n(256-pin)',
            fontsize=5.5, ha='center', va='center', color='#9A3412')

    # Drawing border
    ax.add_patch(plt.Rectangle((-0.5, -2), 14.5, 13,
                               fill=False, ec='#333333', lw=2))

    # Title block
    ax.add_patch(plt.Rectangle((9.5, -2), 4.5, 1, fc='#F9FAFB', ec='#333333', lw=1))
    ax.text(11.75, -1.25, 'DWG: BCI-SEMI-002-B', fontsize=7, ha='center', fontweight='bold')
    ax.text(11.75, -1.65, 'Electrode Array Detail | Rev. 1.0', fontsize=6, ha='center', color='#555')

    plt.tight_layout()
    plt.savefig('BCI_B_electrode_array_detail.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[DWG-B2] Electrode array detail → BCI_B_electrode_array_detail.png")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  BCI Engineering Drawings — Generating...")
    print("=" * 60)
    print()

    draw_invasive_bci_cross_section()
    draw_invasive_bci_exploded()
    draw_semi_invasive_cross_section()
    draw_electrode_array_detail()

    print()
    print("=" * 60)
    print("  Complete. Output files:")
    print("    A1: BCI_A_invasive_cross_section.png")
    print("    A2: BCI_A_invasive_exploded.png")
    print("    B1: BCI_B_semi_invasive_cross_section.png")
    print("    B2: BCI_B_electrode_array_detail.png")
    print("=" * 60)


if __name__ == '__main__':
    main()
