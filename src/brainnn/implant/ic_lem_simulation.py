#!/usr/bin/env python3
"""
IC-LEM: In Silico Closed-Loop Emotional Modulation Simulation
==============================================================
"Beyond the Pill" makalesinde önerilen Closed-Loop Neural Implant teorisini
test etmek için tasarlanmış 3 katmanlı simülasyon iskeleti.

Katman 1 (Wetware)  : Brian2 ile BLA-mPFC nöral devre modeli
Katman 2 (Hardware)  : Phased Array elektrot ve hacimsel iletken E-alan modeli
Katman 3 (Firmware)  : Closed-loop sense → decode → stimulate döngüsü + LSTM

Yazar: IC-LEM Framework
Tarih: 2026-04-01
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt
from brian2 import *

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL PARAMETRELER
# ══════════════════════════════════════════════════════════════════════════════

# Simülasyon
SIM_DURATION     = 2.0 * second   # Toplam simülasyon süresi
DT               = 0.1 * ms       # Zaman adımı
SEED             = 42

# Nöron parametreleri (Izhikevich-tipi basitleştirilmiş LIF)
N_BLA            = 100             # Basolateral Amygdala nöron sayısı
N_MPFC           = 50              # Medial Prefrontal Cortex nöron sayısı

# Sinaptik ağırlıklar
W_BLA_MPFC       = 0.8 * mV       # BLA → mPFC (excitatory, feedforward)
W_MPFC_BLA       = 0.15 * mV      # mPFC → BLA (inhibitory, top-down kontrol)
                                    # ↑ MDD durumunda bu değer ~0.15; sağlıklı ~0.6

# Closed-loop parametreleri
FIRING_THRESHOLD = 60.0            # Hz — patolojik eşik (müdahale tetiklenir)
STIM_CURRENT     = 2.5 * nA       # mPFC'ye uygulanan stimülasyon akımı
SENSE_WINDOW     = 20 * ms        # Ateşleme hızı penceresi

# Gürültü
NOISE_LEVEL      = 0.20           # %20 Gaussian noise

# Elektrot geometri
N_ELECTRODES     = 4               # Phased array elektrot sayısı
GRID_SIZE        = 100             # E-alan hesaplama ızgara boyutu (100x100)
TISSUE_SIGMA     = 0.3             # Doku iletkenliği (S/m)

np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
#  KATMAN 2 — HARDWARE: Phased Array Elektrot ve E-Alan Modeli
# ══════════════════════════════════════════════════════════════════════════════

def compute_single_electrode_field(grid_size, sigma, electrode_pos, current=1.0):
    """
    Tek bir nokta kaynağından (monopolar elektrot) oluşan elektrik alan potansiyelini
    hacimsel iletken modeli ile hesaplar.

    V(r) = I / (4 * pi * sigma * r)

    Parametreler:
        grid_size  : Izgara boyutu (grid_size x grid_size)
        sigma      : Doku iletkenliği (S/m)
        electrode_pos : (x, y) elektrot pozisyonu (mm)
        current    : Uygulanan akım (mA)

    Dönüş:
        V : 2D potansiyel matrisi (mV)
    """
    x = np.linspace(-5, 5, grid_size)  # mm
    y = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x, y)

    ex, ey = electrode_pos
    r = np.sqrt((X - ex)**2 + (Y - ey)**2)
    r = np.maximum(r, 0.05)  # Singülariteyi önle (50 µm minimum)

    V = current / (4 * np.pi * sigma * r)
    return X, Y, V


def compute_phased_array_field(grid_size, sigma, electrode_positions, currents, phases):
    """
    Birden fazla elektrottan gelen akımların süperpozisyonuyla oluşan
    'phased array' girişim desenini hesaplar.

    Her elektrottan gelen potansiyel faz-kaydırılmış olarak toplanır:
    V_total = Σ I_k * cos(φ_k) / (4πσr_k)

    Yapıcı girişim (constructive interference) hedef noktada
    potansiyelin maksimize olmasını sağlar.

    Parametreler:
        electrode_positions : [(x1,y1), (x2,y2), ...] pozisyon listesi
        currents            : Her elektrotun akım genliği
        phases              : Her elektrotun faz açısı (radyan)
    """
    x = np.linspace(-5, 5, grid_size)
    y = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x, y)
    V_total = np.zeros_like(X)

    for (ex, ey), I, phi in zip(electrode_positions, currents, phases):
        r = np.sqrt((X - ex)**2 + (Y - ey)**2)
        r = np.maximum(r, 0.05)
        # Faz-ağırlıklı potansiyel katkısı
        V_total += I * np.cos(phi) / (4 * np.pi * sigma * r)

    return X, Y, V_total


def compute_interference_pattern(grid_size, sigma, target_pos=(0.0, 0.0)):
    """
    4 elektrotun faz açılarını, hedef noktada (BLA basolateral çekirdek)
    yapıcı girişim oluşturacak şekilde hesaplar.

    Strateji: Her elektrotun fazı, sinyalin hedef noktaya ulaştığında
    aynı fazda olması için mesafeye göre ayarlanır.
    """
    # Elektrot pozisyonları — kare dizilim, hedefin etrafında
    electrode_positions = [
        (-3.0, -3.0),  # Sol-alt
        ( 3.0, -3.0),  # Sağ-alt
        (-3.0,  3.0),  # Sol-üst
        ( 3.0,  3.0),  # Sağ-üst
    ]

    # Her elektrottan hedefe olan mesafeyi hesapla
    freq = 130.0  # Hz — DBS frekansı
    wavelength = 1.0 / freq  # Basitleştirilmiş dalga boyu (normalize)

    phases = []
    for (ex, ey) in electrode_positions:
        dist = np.sqrt((target_pos[0] - ex)**2 + (target_pos[1] - ey)**2)
        # Faz: hedefe ulaştığında tüm dalgalar aynı fazda olsun
        phi = -2 * np.pi * dist / (wavelength * 1000)  # Normalize
        phases.append(phi)

    currents = [1.0, 1.0, 1.0, 1.0]  # Eşit akım genliği

    X, Y, V = compute_phased_array_field(
        grid_size, sigma, electrode_positions, currents, phases
    )
    return X, Y, V, electrode_positions


def plot_electrode_comparison():
    """
    Klasik tek elektrot vs. phased array elektrik alan karşılaştırması.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Tek silindirik elektrot
    X, Y, V_single = compute_single_electrode_field(
        GRID_SIZE, TISSUE_SIGMA, (0, 0), current=1.0
    )
    im1 = axes[0].pcolormesh(X, Y, V_single, shading='auto', cmap='hot')
    axes[0].set_title('Klasik Monopolar Elektrot\n(Düşük Spasyal Seçicilik)', fontsize=11)
    axes[0].set_xlabel('x (mm)')
    axes[0].set_ylabel('y (mm)')
    axes[0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0], label='Potansiyel (mV)')
    axes[0].plot(0, 0, 'w+', markersize=15, markeredgewidth=2, label='Elektrot')
    axes[0].legend(loc='upper right', fontsize=8)

    # (b) Phased array — odaklanmış
    X, Y, V_phased, epos = compute_interference_pattern(GRID_SIZE, TISSUE_SIGMA)
    im2 = axes[1].pcolormesh(X, Y, V_phased, shading='auto', cmap='hot')
    axes[1].set_title('Phased Array Elektrot\n(Yüksek Spasyal Seçicilik)', fontsize=11)
    axes[1].set_xlabel('x (mm)')
    axes[1].set_ylabel('y (mm)')
    axes[1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[1], label='Potansiyel (mV)')
    # Elektrot pozisyonlarını göster
    for (ex, ey) in epos:
        axes[1].plot(ex, ey, 'c^', markersize=10, markeredgewidth=1.5,
                     markeredgecolor='white')
    axes[1].plot(0, 0, 'r*', markersize=15, label='Hedef (BLA)')
    axes[1].legend(loc='upper right', fontsize=8)

    # (c) Spasyal seçicilik profili — kesit karşılaştırması
    mid = GRID_SIZE // 2
    profile_single = V_single[mid, :] / np.max(V_single[mid, :])
    profile_phased = V_phased[mid, :] / np.max(V_phased[mid, :])
    x_axis = np.linspace(-5, 5, GRID_SIZE)

    axes[2].plot(x_axis, profile_single, 'r-', linewidth=2, label='Monopolar')
    axes[2].plot(x_axis, profile_phased, 'b-', linewidth=2, label='Phased Array')
    axes[2].axvline(0, color='gray', linestyle='--', alpha=0.5, label='Hedef')
    axes[2].set_title('Spasyal Seçicilik Profili\n(y=0 kesiti, normalize)', fontsize=11)
    axes[2].set_xlabel('x (mm)')
    axes[2].set_ylabel('Normalize Potansiyel')
    axes[2].legend(fontsize=9)
    axes[2].set_xlim(-5, 5)

    plt.tight_layout()
    plt.savefig('electrode_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[HARDWARE] Elektrot karşılaştırma grafiği → electrode_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
#  KATMAN 3 — FIRMWARE: Closed-Loop Sense → Decode → Stimulate
# ══════════════════════════════════════════════════════════════════════════════

class ClosedLoopController:
    """
    Closed-loop nöral modülasyon kontrolcüsü.

    Üç aşamalı döngü:
    1. SENSE   — BLA spike monitor'dan ateşleme hızını oku
    2. DECODE  — Eşik tabanlı + LSTM tahminli karar
    3. STIMULATE — mPFC'ye akım enjekte et → BLA inhibisyon
    """

    def __init__(self, threshold_hz, stim_current, sense_window_ms=20.0,
                 noise_level=0.20, use_lstm=False):
        self.threshold = threshold_hz
        self.stim_current = stim_current
        self.sense_window = sense_window_ms  # ms
        self.noise_level = noise_level
        self.use_lstm = use_lstm

        # Kayıt
        self.firing_rate_log = []
        self.stim_on_log = []
        self.time_log = []
        self.raw_signal_log = []
        self.filtered_signal_log = []

        # Bandpass filtre katsayıları (1-50 Hz, Fs=10000 Hz)
        fs = 1.0 / (float(DT) )  # Örnekleme frekansı
        nyq = fs / 2.0
        low = 1.0 / nyq
        high = min(50.0 / nyq, 0.99)
        if low < high and low > 0:
            self.b_filt, self.a_filt = butter(3, [low, high], btype='band')
        else:
            # Fallback: sadece lowpass
            self.b_filt, self.a_filt = butter(3, min(50.0 / nyq, 0.99), btype='low')

    def sense(self, spike_monitor, current_time_ms, n_neurons):
        """
        BLA nöronlarının anlık ateşleme hızını oku.
        Gaussian gürültü ekle (gerçek beyin sinyali simülasyonu).
        """
        spike_times = np.array(spike_monitor.t / ms)
        window_start = current_time_ms - self.sense_window

        # Pencere içindeki spike sayısı
        spikes_in_window = np.sum(
            (spike_times >= window_start) & (spike_times <= current_time_ms)
        )

        # Ham ateşleme hızı (Hz)
        raw_rate = (spikes_in_window / n_neurons) / (self.sense_window / 1000.0)

        # Gürültü ekle
        noise = np.random.normal(0, self.noise_level * max(raw_rate, 1.0))
        noisy_rate = max(0, raw_rate + noise)

        self.raw_signal_log.append(noisy_rate)

        return noisy_rate, raw_rate

    def decode(self, firing_rate):
        """
        Ateşleme hızını yorumla ve müdahale kararı ver.
        Basit eşik tabanlı decoding.
        """
        stim_on = firing_rate > self.threshold
        return stim_on

    def apply_bandpass_filter(self):
        """
        Kayıtlı ham sinyale bandpass filtre uygula.
        Offline analiz için (simülasyon sonrası karşılaştırma).
        """
        if len(self.raw_signal_log) < 20:
            return self.raw_signal_log.copy()

        try:
            filtered = filtfilt(self.b_filt, self.a_filt,
                               self.raw_signal_log, padlen=min(10, len(self.raw_signal_log)-1))
            return filtered.tolist()
        except Exception:
            return self.raw_signal_log.copy()


# ══════════════════════════════════════════════════════════════════════════════
#  KATMAN 1 — WETWARE: Brian2 BLA-mPFC Nöral Devre
# ══════════════════════════════════════════════════════════════════════════════

def run_neural_simulation():
    """
    Brian2 ile BLA ↔ mPFC kapalı-döngü nöral devre simülasyonu.

    Devre Yapısı:
    ─────────────
    BLA (N=100, eksitatör LIF nöronlar)
      │
      │ excitatory (feedforward)
      ▼
    mPFC (N=50, inhibitör LIF nöronlar)
      │
      │ inhibitory (top-down feedback)
      ▼
    BLA (ateşleme baskılanır)

    MDD Modeli: W_MPFC_BLA düşük → top-down kontrol zayıf → BLA hiperaktif
    Tedavi: Closed-loop stimülasyon → mPFC güçlendirilir → BLA normalize
    """

    start_scope()
    prefs.codegen.target = 'numpy'
    defaultclock.dt = DT

    # ── LIF Nöron Modeli ──
    eqs_excitatory = '''
    dv/dt = (-(v - v_rest) + R*I_ext + R*I_syn + R*I_stim) / tau : volt
    I_ext  : amp    # Dışsal giriş (patolojik sürücü)
    I_syn  : amp    # Sinaptik akım
    I_stim : amp    # Closed-loop stimülasyon akımı
    v_rest : volt
    R      : ohm
    tau    : second
    '''

    eqs_inhibitory = '''
    dv/dt = (-(v - v_rest) + R*I_ext + R*I_syn + R*I_stim) / tau : volt
    I_ext  : amp
    I_syn  : amp
    I_stim : amp
    v_rest : volt
    R      : ohm
    tau    : second
    '''

    # ── BLA Nöronları (Eksitatör) ──
    bla = NeuronGroup(
        N_BLA, eqs_excitatory,
        threshold='v > -50*mV',
        reset='v = -65*mV',
        refractory=2*ms,
        method='euler'
    )
    bla.v = -65*mV
    bla.v_rest = -65*mV
    bla.R = 100*Mohm
    bla.tau = 10*ms
    # Patolojik durum: BLA'ya yüksek tonic giriş (korku/anksiyete sürücüsü)
    bla.I_ext = '(0.3 + 0.15*rand()) * nA'  # Heterojen, yüksek sürücü
    bla.I_stim = 0*nA

    # ── mPFC Nöronları (İnhibitör feedback) ──
    mpfc = NeuronGroup(
        N_MPFC, eqs_inhibitory,
        threshold='v > -50*mV',
        reset='v = -65*mV',
        refractory=2*ms,
        method='euler'
    )
    mpfc.v = -65*mV
    mpfc.v_rest = -65*mV
    mpfc.R = 100*Mohm
    mpfc.tau = 20*ms  # mPFC daha yavaş dinamik
    mpfc.I_ext = '0.05 * nA'  # Düşük bazal aktivite
    mpfc.I_stim = 0*nA

    # ── Sinaptik Bağlantılar ──
    # BLA → mPFC (excitatory feedforward)
    syn_bla_mpfc = Synapses(bla, mpfc, on_pre='v_post += w',
                            namespace={'w': W_BLA_MPFC})
    syn_bla_mpfc.connect(p=0.3)  # %30 bağlantı olasılığı

    # mPFC → BLA (inhibitory top-down feedback)
    # MDD durumu: w düşük (0.15 mV) → yetersiz inhibisyon
    syn_mpfc_bla = Synapses(mpfc, bla, on_pre='v_post -= w',
                            namespace={'w': W_MPFC_BLA})
    syn_mpfc_bla.connect(p=0.4)

    # BLA içi rekürrent excitatory bağlantılar (amygdala reverberasyon)
    syn_bla_bla = Synapses(bla, bla, on_pre='v_post += w',
                           namespace={'w': 0.3 * mV})
    syn_bla_bla.connect(p=0.1, condition='i != j')

    # ── Monitörler ──
    bla_spikemon = SpikeMonitor(bla)
    mpfc_spikemon = SpikeMonitor(mpfc)
    bla_ratemon = PopulationRateMonitor(bla)
    mpfc_ratemon = PopulationRateMonitor(mpfc)

    # ══════════════════════════════════════════════════════════════════════
    #  CLOSED-LOOP ÇALIŞMA DÖNGÜSÜ
    # ══════════════════════════════════════════════════════════════════════

    controller = ClosedLoopController(
        threshold_hz=FIRING_THRESHOLD,
        stim_current=STIM_CURRENT,
        sense_window_ms=float(SENSE_WINDOW / ms),
        noise_level=NOISE_LEVEL
    )

    # Simülasyonu parçalara bölerek closed-loop'u gerçekleştir
    # (Brian2'de run() sırasında Python müdahalesi için segment bazlı çalıştırma)

    total_time_ms = float(SIM_DURATION / ms)
    step_ms = float(SENSE_WINDOW / ms)  # Her 20 ms'de bir kontrol
    current_ms = 0.0

    print("[WETWARE] Simülasyon başlıyor...")
    print(f"  BLA nöron: {N_BLA}, mPFC nöron: {N_MPFC}")
    print(f"  MDD modeli: W_mPFC→BLA = {W_MPFC_BLA} (sağlıklı ~0.6 mV)")
    print(f"  Eşik: {FIRING_THRESHOLD} Hz, Stimülasyon: {STIM_CURRENT}")
    print(f"  Gürültü seviyesi: {NOISE_LEVEL*100:.0f}%")
    print()

    # İlk 500 ms: Open-loop (stimülasyon YOK) — patolojik bazal durum
    # 500-2000 ms: Closed-loop AKTİF
    CLOSED_LOOP_START = 500.0  # ms

    while current_ms < total_time_ms:
        # Brian2'de bir adım çalıştır
        run(step_ms * ms)
        current_ms += step_ms

        # SENSE — BLA ateşleme hızını oku
        noisy_rate, raw_rate = controller.sense(
            bla_spikemon, current_ms, N_BLA
        )

        # Closed-loop aktif mi?
        if current_ms >= CLOSED_LOOP_START:
            # DECODE — müdahale gerekli mi?
            stim_on = controller.decode(noisy_rate)

            # STIMULATE — mPFC'yi uyar, BLA'yı baskıla
            if stim_on:
                mpfc.I_stim = STIM_CURRENT
                # BLA'ya doğrudan hafif inhibitör akım (elektrot etkisi)
                bla.I_stim = -0.5 * nA
            else:
                mpfc.I_stim = 0 * nA
                bla.I_stim = 0 * nA
        else:
            stim_on = False

        controller.firing_rate_log.append(raw_rate)
        controller.stim_on_log.append(1.0 if stim_on else 0.0)
        controller.time_log.append(current_ms)

    print(f"[WETWARE] Simülasyon tamamlandı: {total_time_ms:.0f} ms")

    # Filtrelenmiş sinyal
    controller.filtered_signal_log = controller.apply_bandpass_filter()

    return (bla_spikemon, mpfc_spikemon, bla_ratemon, mpfc_ratemon,
            controller)


# ══════════════════════════════════════════════════════════════════════════════
#  GÖRSELLEŞTİRME
# ══════════════════════════════════════════════════════════════════════════════

def plot_neural_results(bla_spikemon, mpfc_spikemon, bla_ratemon,
                        mpfc_ratemon, controller):
    """
    3 panelli sonuç grafiği:
    (a) Raster plot — BLA ve mPFC spike zamanları
    (b) Line chart — BLA firing rate + stimülasyon zamanları
    (c) Raw vs Filtered sinyal karşılaştırması
    """

    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(4, 1, height_ratios=[1.2, 1, 0.8, 0.8], hspace=0.35)

    closed_loop_start = 500.0  # ms

    # ── (a) Raster Plot ──
    ax1 = fig.add_subplot(gs[0])

    # BLA spikes (kırmızı)
    bla_t = np.array(bla_spikemon.t / ms)
    bla_i = np.array(bla_spikemon.i)
    ax1.scatter(bla_t, bla_i, s=0.5, c='crimson', alpha=0.6, label='BLA (excitatory)')

    # mPFC spikes (mavi, BLA'nın üstünde offset)
    mpfc_t = np.array(mpfc_spikemon.t / ms)
    mpfc_i = np.array(mpfc_spikemon.i) + N_BLA  # Offset
    ax1.scatter(mpfc_t, mpfc_i, s=0.5, c='dodgerblue', alpha=0.6, label='mPFC (inhibitory)')

    ax1.axvline(closed_loop_start, color='green', linestyle='--', linewidth=2,
                label='Closed-Loop Başlangıcı')
    ax1.axhline(N_BLA, color='gray', linestyle='-', alpha=0.3)
    ax1.set_ylabel('Nöron İndeksi')
    ax1.set_title('(a) Raster Plot — BLA ve mPFC Nöral Aktivite', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, markerscale=8)
    ax1.set_xlim(0, float(SIM_DURATION / ms))

    # ── (b) BLA Ateşleme Hızı + Stimülasyon ──
    ax2 = fig.add_subplot(gs[1])

    times = np.array(controller.time_log)
    rates = np.array(controller.firing_rate_log)
    stim = np.array(controller.stim_on_log)

    ax2.plot(times, rates, 'crimson', linewidth=1.5, label='BLA Firing Rate (Hz)')
    ax2.axhline(FIRING_THRESHOLD, color='orange', linestyle='--', linewidth=1.5,
                label=f'Eşik ({FIRING_THRESHOLD} Hz)')
    ax2.axvline(closed_loop_start, color='green', linestyle='--', linewidth=2)

    # Stimülasyon aktif bölgeleri yeşil ile göster
    stim_times = times[stim > 0.5]
    if len(stim_times) > 0:
        for t in stim_times:
            ax2.axvspan(t - 10, t + 10, alpha=0.15, color='green')
        # Legend için tek bir patch
        ax2.fill_between([], [], alpha=0.3, color='green', label='Stimülasyon Aktif')

    ax2.set_ylabel('Ateşleme Hızı (Hz)')
    ax2.set_title('(b) BLA Aktivitesi ve Closed-Loop Müdahale', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(0, float(SIM_DURATION / ms))

    # ── (c) Ham vs Filtrelenmiş Sinyal ──
    ax3 = fig.add_subplot(gs[2])

    raw = np.array(controller.raw_signal_log)
    filtered = np.array(controller.filtered_signal_log)

    ax3.plot(times, raw, 'gray', alpha=0.5, linewidth=0.8, label='Ham Sinyal (gürültülü)')
    if len(filtered) == len(times):
        ax3.plot(times, filtered, 'navy', linewidth=1.5, label='Bandpass Filtrelenmiş')
    ax3.axhline(FIRING_THRESHOLD, color='orange', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Ateşleme Hızı (Hz)')
    ax3.set_xlabel('Zaman (ms)')
    ax3.set_title('(c) Ham vs Filtrelenmiş Sinyal — Gürültü Etkisi', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_xlim(0, float(SIM_DURATION / ms))

    # ── (d) Stimülasyon Zamanlama ──
    ax4 = fig.add_subplot(gs[3])
    ax4.fill_between(times, 0, stim, step='mid', alpha=0.7, color='green',
                     label='Stimülasyon ON/OFF')
    ax4.set_ylabel('Stim.')
    ax4.set_xlabel('Zaman (ms)')
    ax4.set_title('(d) Stimülasyon Zamanlama Diyagramı', fontsize=13, fontweight='bold')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['OFF', 'ON'])
    ax4.set_xlim(0, float(SIM_DURATION / ms))
    ax4.legend(loc='upper right', fontsize=9)

    plt.savefig('neural_simulation_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[VISUAL] Nöral simülasyon grafikleri → neural_simulation_results.png")


def plot_noise_comparison(controller):
    """
    Gürültülü (raw) vs filtrelenmiş sinyalin decoding performans karşılaştırması.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    times = np.array(controller.time_log)
    raw = np.array(controller.raw_signal_log)
    filtered = np.array(controller.filtered_signal_log)
    true_rates = np.array(controller.firing_rate_log)

    # Ham sinyalden decode
    raw_decisions = (raw > FIRING_THRESHOLD).astype(float)
    # Filtrelenmiş sinyalden decode
    if len(filtered) == len(times):
        filt_decisions = (filtered > FIRING_THRESHOLD).astype(float)
    else:
        filt_decisions = raw_decisions.copy()

    # Gerçek durum (gürültüsüz)
    true_decisions = (true_rates > FIRING_THRESHOLD).astype(float)

    # Performans metrikleri
    cl_start_idx = np.searchsorted(times, 500.0)

    if len(times) > cl_start_idx:
        raw_accuracy = np.mean(raw_decisions[cl_start_idx:] == true_decisions[cl_start_idx:])
        filt_accuracy = np.mean(filt_decisions[cl_start_idx:] == true_decisions[cl_start_idx:])
    else:
        raw_accuracy = 0
        filt_accuracy = 0

    # (a) Karar karşılaştırması
    axes[0].plot(times, true_decisions * 1.0, 'k-', linewidth=2, alpha=0.4,
                 label=f'Gerçek Durum')
    axes[0].plot(times, raw_decisions * 0.95, 'r.', markersize=2, alpha=0.5,
                 label=f'Ham Sinyal ({raw_accuracy:.1%} doğruluk)')
    axes[0].plot(times, filt_decisions * 1.05, 'b.', markersize=2, alpha=0.5,
                 label=f'Filtrelenmiş ({filt_accuracy:.1%} doğruluk)')
    axes[0].set_title('Decoding Kararları: Ham vs Filtrelenmiş', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Zaman (ms)')
    axes[0].set_ylabel('Karar (0=Normal, 1=Patolojik)')
    axes[0].legend(fontsize=9)

    # (b) Hata dağılımı
    raw_errors = np.abs(raw[cl_start_idx:] - true_rates[cl_start_idx:])
    filt_errors = np.abs(filtered[cl_start_idx:] - true_rates[cl_start_idx:]) if len(filtered) == len(times) else raw_errors

    axes[1].hist(raw_errors, bins=30, alpha=0.5, color='red', label='Ham Sinyal Hatası')
    axes[1].hist(filt_errors, bins=30, alpha=0.5, color='blue', label='Filtrelenmiş Sinyal Hatası')
    axes[1].set_title('Decoding Hata Dağılımı', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Mutlak Hata (Hz)')
    axes[1].set_ylabel('Frekans')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('noise_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VISUAL] Gürültü karşılaştırma → noise_comparison.png")
    print(f"  Ham sinyal doğruluğu:   {raw_accuracy:.1%}")
    print(f"  Filtrelenmiş doğruluk:  {filt_accuracy:.1%}")


# ══════════════════════════════════════════════════════════════════════════════
#  ANA ÇALIŞMA AKIŞI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    IC-LEM simülasyonunun ana çalışma akışı.
    """
    print("=" * 70)
    print("  IC-LEM: In Silico Closed-Loop Emotional Modulation Simulation")
    print("  'Beyond the Pill' — Closed-Loop Neural Implant Teorisi Testi")
    print("=" * 70)
    print()

    # ── KATMAN 2: Elektrot E-alan analizi ──
    print("─" * 50)
    print("  KATMAN 2 — HARDWARE: Elektrot E-Alan Analizi")
    print("─" * 50)
    plot_electrode_comparison()
    print()

    # ── KATMAN 1 + 3: Nöral devre + Closed-loop ──
    print("─" * 50)
    print("  KATMAN 1+3 — WETWARE + FIRMWARE: Nöral Simülasyon")
    print("─" * 50)
    (bla_spikemon, mpfc_spikemon, bla_ratemon,
     mpfc_ratemon, controller) = run_neural_simulation()
    print()

    # ── Görselleştirme ──
    print("─" * 50)
    print("  GÖRSELLEŞTİRME")
    print("─" * 50)
    plot_neural_results(bla_spikemon, mpfc_spikemon,
                       bla_ratemon, mpfc_ratemon, controller)
    plot_noise_comparison(controller)

    # ── Özet istatistikler ──
    print()
    print("─" * 50)
    print("  ÖZET İSTATİSTİKLER")
    print("─" * 50)

    times = np.array(controller.time_log)
    rates = np.array(controller.firing_rate_log)
    stim = np.array(controller.stim_on_log)

    # İlk 500ms (open-loop) vs son 500ms (closed-loop)
    ol_mask = times < 500.0
    cl_mask = times > 1500.0

    if np.any(ol_mask) and np.any(cl_mask):
        ol_mean = np.mean(rates[ol_mask])
        cl_mean = np.mean(rates[cl_mask])
        reduction = ((ol_mean - cl_mean) / ol_mean) * 100

        print(f"  Open-loop BLA ort. ateşleme:   {ol_mean:.1f} Hz")
        print(f"  Closed-loop BLA ort. ateşleme: {cl_mean:.1f} Hz")
        print(f"  Aktivite azalması:             {reduction:.1f}%")
        print(f"  Stimülasyon oranı:             {np.mean(stim[~ol_mask]):.1%}")

    total_stim_events = int(np.sum(stim))
    print(f"  Toplam stimülasyon olayı:      {total_stim_events}")
    print()
    print("=" * 70)
    print("  Simülasyon tamamlandı. Çıktı dosyaları:")
    print("    1. electrode_comparison.png     — Monopolar vs Phased Array")
    print("    2. neural_simulation_results.png — Raster + Aktivite + Stim")
    print("    3. noise_comparison.png          — Ham vs Filtrelenmiş sinyal")
    print("=" * 70)


if __name__ == '__main__':
    main()
