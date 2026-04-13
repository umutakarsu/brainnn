"""
IC-LEM Dashboard — Multilingual Translation Module
====================================================
Academic-grade translations: Turkish (TR), English (EN), German (DE)

All neuroscience and pharmacological terminology follows
peer-reviewed conventions (APA, DGN, TND standards).
"""

T = {
    # ── Language selector ──
    "lang_label": {
        "tr": "🌐 Dil",
        "en": "🌐 Language",
        "de": "🌐 Sprache",
    },

    # ── Page header ──
    "page_title": {
        "tr": "IC-LEM Simülasyon Dashboard",
        "en": "IC-LEM Simulation Dashboard",
        "de": "IC-LEM Simulationsdashboard",
    },
    "page_subtitle": {
        "tr": "In Silico Closed-Loop Emotional Modulation",
        "en": "In Silico Closed-Loop Emotional Modulation",
        "de": "In-Silico-Closed-Loop-Emotionsmodulation",
    },
    "page_desc": {
        "tr": '"Beyond the Pill" — İlaç vs. Nöral İmplant Karşılaştırmalı Simülasyon Platformu',
        "en": '"Beyond the Pill" — Comparative Drug vs. Neural Implant Simulation Platform',
        "de": '"Beyond the Pill" — Vergleichende Simulations\u00ADplattform: Pharmakotherapie vs. neuronales Implantat',
    },

    # ── Sidebar ──
    "sidebar_title": {
        "tr": "Simülasyon Kontrolleri",
        "en": "Simulation Controls",
        "de": "Simulationsparameter",
    },
    "mode_label": {
        "tr": "🔬 Müdahale Modu",
        "en": "🔬 Intervention Mode",
        "de": "🔬 Interventionsmodus",
    },
    "mode_help": {
        "tr": "Karşılaştırmak istediğiniz tedavi yöntemini seçin.",
        "en": "Select the therapeutic modality for comparison.",
        "de": "Wählen Sie die therapeutische Modalität zum Vergleich.",
    },
    "bio_params_title": {
        "tr": "🧬 Biyolojik Parametreler",
        "en": "🧬 Biological Parameters",
        "de": "🧬 Biologische Parameter",
    },
    "w_mpfc_bla_label": {
        "tr": "mPFC → BLA Sinaptik Ağırlık (mV)",
        "en": "mPFC → BLA Synaptic Weight (mV)",
        "de": "mPFC → BLA Synaptisches Gewicht (mV)",
    },
    "w_mpfc_bla_help": {
        "tr": "MDD'de düşük (~0.15), sağlıklıda yüksek (~0.60). Top-down kortiko-limbik kontrol gücü.",
        "en": "Low in MDD (~0.15), healthy (~0.60). Top-down cortico-limbic regulatory strength.",
        "de": "Niedrig bei MDD (~0,15), gesund (~0,60). Top-down-kortikolimbische Regulationsstärke.",
    },
    "noise_label": {
        "tr": "Gürültü Seviyesi (%)",
        "en": "Noise Level (%)",
        "de": "Rauschpegel (%)",
    },
    "noise_help": {
        "tr": "Gaussian gürültü — gerçek beyin sinyali gürültüsünü simüle eder.",
        "en": "Gaussian noise — simulates realistic neural signal noise.",
        "de": "Gauß-Rauschen — simuliert realistisches neuronales Signalrauschen.",
    },
    "cl_params_title": {
        "tr": "🎛️ Kapalı Döngü Parametreleri",
        "en": "🎛️ Closed-Loop Parameters",
        "de": "🎛️ Closed-Loop-Parameter",
    },
    "threshold_label": {
        "tr": "Müdahale Eşiği (Hz)",
        "en": "Intervention Threshold (Hz)",
        "de": "Interventionsschwelle (Hz)",
    },
    "threshold_help": {
        "tr": "BLA ateşleme hızı bu değeri aştığında stimülasyon tetiklenir.",
        "en": "Stimulation is triggered when BLA firing rate exceeds this value.",
        "de": "Stimulation wird ausgelöst, wenn die BLA-Feuerungsrate diesen Wert überschreitet.",
    },
    "stim_label": {
        "tr": "Stimülasyon Akımı (nA)",
        "en": "Stimulation Current (nA)",
        "de": "Stimulationsstrom (nA)",
    },
    "stim_help": {
        "tr": "mPFC'ye uygulanan stimülasyon akım şiddeti.",
        "en": "Stimulation current intensity applied to mPFC.",
        "de": "Stimulationsstromstärke, die auf den mPFC angewendet wird.",
    },
    "sim_title": {
        "tr": "Simülasyon",
        "en": "Simulation",
        "de": "Simulation",
    },
    "duration_label": {
        "tr": "Süre (ms)",
        "en": "Duration (ms)",
        "de": "Dauer (ms)",
    },
    "run_button": {
        "tr": "▶️ Simülasyonu Başlat",
        "en": "▶️ Run Simulation",
        "de": "▶️ Simulation starten",
    },
    "running_spinner": {
        "tr": "simülasyonu çalışıyor...",
        "en": "simulation running...",
        "de": "Simulation läuft...",
    },
    "info_msg": {
        "tr": "☝️ Yukarıdan bir tedavi yöntemi seçerek simülasyonu başlatın.",
        "en": "☝️ Select a treatment above to start the simulation.",
        "de": "☝️ Wählen Sie oben eine Behandlung aus, um die Simulation zu starten.",
    },

    # ── Drug descriptions (short) ──
    "desc_ssri": {
        "tr": "En yaygın antidepresan. Beyindeki serotonin seviyesini artırır, ancak etkisi haftalarca sürer ve tüm beyni etkiler — sadece sorunlu bölgeyi değil.",
        "en": "The most common antidepressant. It raises serotonin levels in the brain, but takes weeks to work and affects the entire brain — not just the problem area.",
        "de": "Das häufigste Antidepressivum. Es erhöht den Serotoninspiegel im Gehirn, braucht aber Wochen und wirkt auf das gesamte Gehirn — nicht nur auf den Problembereich.",
    },
    "desc_benzo": {
        "tr": "Anksiyete ve panik atak için sık kullanılan sakinleştirici. Çok hızlı etki eder ama bağımlılık riski yüksektir ve düşünme hızını yavaşlatır.",
        "en": "A widely used sedative for anxiety and panic attacks. Very fast-acting, but carries high dependence risk and slows cognitive processing.",
        "de": "Ein weit verbreitetes Beruhigungsmittel gegen Angst und Panikattacken. Sehr schnell wirksam, aber mit hohem Abhängigkeitsrisiko und kognitiver Verlangsamung.",
    },
    "desc_ketamine": {
        "tr": "Yeni nesil hızlı etkili antidepresan. Saatler içinde yeni sinaptik bağlantılar kurar, ancak etkisi geçicidir — tekrar tekrar uygulanması gerekir.",
        "en": "A next-generation, fast-acting antidepressant. It builds new synaptic connections within hours, but the effect is temporary — repeated doses are needed.",
        "de": "Ein Antidepressivum der neuen Generation mit schnellem Wirkungseintritt. Es bildet innerhalb von Stunden neue synaptische Verbindungen, die Wirkung ist jedoch vorübergehend.",
    },
    "desc_iclem": {
        "tr": "Beyne yerleştirilen küçük bir çip, sadece sorunlu devreyi algılayıp milisaniyeler içinde düzeltir — ilaç gibi tüm beyni etkilemez.",
        "en": "A small chip implanted in the brain detects and corrects only the disrupted circuit within milliseconds — unlike medication, it does not affect the entire brain.",
        "de": "Ein kleiner, ins Gehirn implantierter Chip erkennt und korrigiert nur den gestörten Schaltkreis innerhalb von Millisekunden — anders als Medikamente beeinflusst er nicht das gesamte Gehirn.",
    },

    # ── Drug mechanisms ──
    "mech_ssri": {
        "tr": "Seçici Serotonin Geri Alım İnhibisyonu",
        "en": "Selective Serotonin Reuptake Inhibition",
        "de": "Selektive Serotonin-Wiederaufnahmehemmung",
    },
    "mech_benzo": {
        "tr": "GABA-A Reseptör Pozitif Allosterik Modülatörü",
        "en": "GABA-A Receptor Positive Allosteric Modulator",
        "de": "GABA-A-Rezeptor Positiver Allosterischer Modulator",
    },
    "mech_ketamine": {
        "tr": "NMDA Reseptör Antagonizmi / Hızlı Sinaptogenez",
        "en": "NMDA Receptor Antagonism / Rapid Synaptogenesis",
        "de": "NMDA-Rezeptor-Antagonismus / Schnelle Synaptogenese",
    },
    "mech_iclem": {
        "tr": "Kapalı Döngü Nöromodülasyon (Phased Array Stimülasyon)",
        "en": "Closed-Loop Neuromodulation (Phased Array Stimulation)",
        "de": "Closed-Loop-Neuromodulation (Phased-Array-Stimulation)",
    },

    # ── Top metrics ──
    "metric_bla_reduction": {
        "tr": "BLA Aktivite Azalması",
        "en": "BLA Activity Reduction",
        "de": "BLA-Aktivitätsreduktion",
    },
    "metric_stim_ratio": {
        "tr": "Stimülasyon Oranı",
        "en": "Stimulation Duty Cycle",
        "de": "Stimulationsquote",
    },
    "metric_nac_blunting": {
        "tr": "NAc Körelme",
        "en": "NAc Blunting",
        "de": "NAc-Abstumpfung",
    },
    "metric_onset": {
        "tr": "Etki Hızı",
        "en": "Onset Latency",
        "de": "Wirkungseintritt",
    },
    "metric_precision": {
        "tr": "Spasyal Hassasiyet",
        "en": "Spatial Selectivity",
        "de": "Räumliche Selektivität",
    },

    # ── Onset latency values ──
    "onset_ssri": {
        "tr": "2-4 Hafta", "en": "2–4 Weeks", "de": "2–4 Wochen",
    },
    "onset_benzo": {
        "tr": "< 30 dk", "en": "< 30 min", "de": "< 30 Min.",
    },
    "onset_ketamine": {
        "tr": "< 2 Saat", "en": "< 2 Hours", "de": "< 2 Stunden",
    },
    "onset_iclem": {
        "tr": "< 100 ms", "en": "< 100 ms", "de": "< 100 ms",
    },

    # ── Tabs ──
    "tab_neural": {
        "tr": "📊 Nöral Aktivite",
        "en": "📊 Neural Activity",
        "de": "📊 Neuronale Aktivität",
    },
    "tab_efield": {
        "tr": "⚡ Elektrik Alan",
        "en": "⚡ Electric Field",
        "de": "⚡ Elektrisches Feld",
    },
    "tab_comparison": {
        "tr": "📋 Karşılaştırma Tablosu",
        "en": "📋 Comparative Table",
        "de": "📋 Vergleichstabelle",
    },
    "tab_analysis": {
        "tr": "📝 Analiz Raporu",
        "en": "📝 Analysis Report",
        "de": "📝 Analysebericht",
    },

    # ── Neural Activity plots ──
    "plot_raster_title": {
        "tr": "Raster Plot — BLA Nöral Ateşleme",
        "en": "Raster Plot — BLA Neural Firing",
        "de": "Raster-Plot — BLA Neuronale Feuerung",
    },
    "plot_bla_rate_title": {
        "tr": "BLA Amygdala Aktivitesi (Hz)",
        "en": "BLA Amygdala Activity (Hz)",
        "de": "BLA-Amygdala-Aktivität (Hz)",
    },
    "plot_nac_title": {
        "tr": "Nucleus Accumbens (Ödül Devresi) Aktivitesi",
        "en": "Nucleus Accumbens (Reward Circuit) Activity",
        "de": "Nucleus Accumbens (Belohnungsschaltkreis) Aktivität",
    },
    "plot_stim_title": {
        "tr": "Stimülasyon Zamanlaması",
        "en": "Stimulation Timing",
        "de": "Stimulationszeitgebung",
    },
    "legend_bla_spikes": {
        "tr": "BLA Spike'ları", "en": "BLA Spikes", "de": "BLA-Spikes",
    },
    "legend_mpfc_spikes": {
        "tr": "mPFC Spike'ları", "en": "mPFC Spikes", "de": "mPFC-Spikes",
    },
    "legend_cl_on": {
        "tr": "Kapalı Döngü AÇIK", "en": "Closed-Loop ON", "de": "Closed-Loop EIN",
    },
    "legend_bla_rate": {
        "tr": "BLA Ateşleme Hızı (Hz)", "en": "BLA Firing Rate (Hz)", "de": "BLA-Feuerungsrate (Hz)",
    },
    "legend_threshold": {
        "tr": "Eşik", "en": "Threshold", "de": "Schwelle",
    },
    "legend_stim_active": {
        "tr": "Stimülasyon Aktif", "en": "Stimulation Active", "de": "Stimulation aktiv",
    },
    "legend_nac_rate": {
        "tr": "NAc Ateşleme Hızı", "en": "NAc Firing Rate", "de": "NAc-Feuerungsrate",
    },
    "legend_stim_onoff": {
        "tr": "Stim AÇ/KAPA", "en": "Stim ON/OFF", "de": "Stim EIN/AUS",
    },
    "axis_neuron": {
        "tr": "Nöron #", "en": "Neuron #", "de": "Neuron Nr.",
    },
    "axis_hz": {"tr": "Hz", "en": "Hz", "de": "Hz"},
    "axis_onoff": {"tr": "AÇ/KAPA", "en": "ON/OFF", "de": "EIN/AUS"},
    "axis_time": {"tr": "Zaman (ms)", "en": "Time (ms)", "de": "Zeit (ms)"},

    # ── E-field tab ──
    "efield_title": {
        "tr": "Phased Array vs Monopolar Elektrot — Elektrik Alan Dağılımı",
        "en": "Phased Array vs Monopolar Electrode — Electric Field Distribution",
        "de": "Phased Array vs. monopolare Elektrode — Elektrische Feldverteilung",
    },
    "efield_mono_title": {
        "tr": "Klasik Monopolar Elektrot",
        "en": "Conventional Monopolar Electrode",
        "de": "Konventionelle monopolare Elektrode",
    },
    "efield_mono_sub": {
        "tr": "Düşük Spasyal Seçicilik",
        "en": "Low Spatial Selectivity",
        "de": "Niedrige räumliche Selektivität",
    },
    "efield_phased_title": {
        "tr": "Phased Array Elektrot",
        "en": "Phased Array Electrode",
        "de": "Phased-Array-Elektrode",
    },
    "efield_phased_sub": {
        "tr": "Yüksek Spasyal Seçicilik — Yapıcı Girişim",
        "en": "High Spatial Selectivity — Constructive Interference",
        "de": "Hohe räumliche Selektivität — Konstruktive Interferenz",
    },
    "efield_profile_title": {
        "tr": "Spasyal Seçicilik Profili (y=0 Kesiti)",
        "en": "Spatial Selectivity Profile (y=0 Cross-Section)",
        "de": "Räumliches Selektivitätsprofil (y=0-Querschnitt)",
    },
    "efield_legend_electrodes": {
        "tr": "Elektrotlar", "en": "Electrodes", "de": "Elektroden",
    },
    "efield_legend_target": {
        "tr": "Hedef (BLA)", "en": "Target (BLA)", "de": "Ziel (BLA)",
    },
    "efield_legend_mono": {
        "tr": "Monopolar", "en": "Monopolar", "de": "Monopolar",
    },
    "efield_legend_phased": {
        "tr": "Phased Array", "en": "Phased Array", "de": "Phased Array",
    },
    "efield_legend_target_line": {
        "tr": "Hedef", "en": "Target", "de": "Ziel",
    },
    "efield_axis_norm": {
        "tr": "Normalize Potansiyel",
        "en": "Normalized Potential",
        "de": "Normiertes Potential",
    },

    # ── Comparison table ──
    "comp_title": {
        "tr": "📊 Karşılaştırmalı Performans Tablosu",
        "en": "📊 Comparative Performance Table",
        "de": "📊 Vergleichende Leistungstabelle",
    },
    "comp_subtitle": {
        "tr": "Makale Results bölümü için — simülasyon verilerine dayalı",
        "en": "For the Results section — based on simulation data",
        "de": "Für den Ergebnisteil — basierend auf Simulationsdaten",
    },
    "comp_numerical": {
        "tr": "Sayısal Özet",
        "en": "Numerical Summary",
        "de": "Numerische Zusammenfassung",
    },
    "comp_score": {
        "tr": "Genel Skor", "en": "Overall Score", "de": "Gesamtbewertung",
    },

    # ── Table row labels ──
    "tbl_param": {"tr": "Parametre", "en": "Parameter", "de": "Parameter"},
    "tbl_onset": {"tr": "Etki Hızı", "en": "Onset Latency", "de": "Wirkungseintritt"},
    "tbl_spatial": {
        "tr": "Spasyal Hassasiyet", "en": "Spatial Selectivity", "de": "Räumliche Selektivität",
    },
    "tbl_bla_ctrl": {
        "tr": "BLA Aktivite Kontrolü", "en": "BLA Activity Control", "de": "BLA-Aktivitätskontrolle",
    },
    "tbl_blunting": {
        "tr": "Duygusal Küntleşme Riski",
        "en": "Emotional Blunting Risk",
        "de": "Risiko Emotionaler Abstumpfung",
    },
    "tbl_cognitive": {
        "tr": "Bilişsel Yan Etki",
        "en": "Cognitive Side Effects",
        "de": "Kognitive Nebenwirkungen",
    },
    "tbl_addiction": {
        "tr": "Bağımlılık Riski", "en": "Dependence Risk", "de": "Abhängigkeitsrisiko",
    },
    "tbl_targeting": {
        "tr": "Hedefleme Yöntemi", "en": "Targeting Method", "de": "Zielverfahren",
    },
    "tbl_dosing": {
        "tr": "Doz Ayarlanabilirliği", "en": "Dose Titratability", "de": "Dosistitrierbarkeit",
    },
    "tbl_reversibility": {
        "tr": "Geri Dönüşümlülük", "en": "Reversibility", "de": "Reversibilität",
    },
    "tbl_longterm": {
        "tr": "Uzun Vadeli Etkinlik", "en": "Long-Term Efficacy", "de": "Langzeitwirksamkeit",
    },

    # ── Table SSRI column ──
    "ssri_onset": {"tr": "2-4 Hafta", "en": "2–4 Weeks", "de": "2–4 Wochen"},
    "ssri_spatial": {
        "tr": "~%10 (Tüm beyin)", "en": "~10% (Whole brain)", "de": "~10 % (Gesamtes Gehirn)",
    },
    "ssri_bla": {
        "tr": "Orta (%30-50 azalma)", "en": "Moderate (30–50% reduction)",
        "de": "Mäßig (30–50 % Reduktion)",
    },
    "ssri_blunting": {"tr": "🔴 Yüksek", "en": "🔴 High", "de": "🔴 Hoch"},
    "ssri_cognitive": {"tr": "🟡 Brain Fog", "en": "🟡 Brain Fog", "de": "🟡 Brain Fog"},
    "ssri_addiction": {"tr": "🟢 Düşük", "en": "🟢 Low", "de": "🟢 Niedrig"},
    "ssri_targeting": {"tr": "Sistemik (oral)", "en": "Systemic (oral)", "de": "Systemisch (oral)"},
    "ssri_dosing": {
        "tr": "Kaba (mg cinsinden)", "en": "Coarse (mg scale)", "de": "Grob (mg-Skala)",
    },
    "ssri_reversibility": {
        "tr": "Haftalarca sürer", "en": "Weeks to wash out", "de": "Wochen bis zur Auswaschung",
    },
    "ssri_longterm": {"tr": "🟡 Orta", "en": "🟡 Moderate", "de": "🟡 Mäßig"},

    # ── Table Benzo column ──
    "benzo_onset": {"tr": "< 30 Dakika", "en": "< 30 Minutes", "de": "< 30 Minuten"},
    "benzo_spatial": {
        "tr": "~%5 (Tüm beyin)", "en": "~5% (Whole brain)", "de": "~5 % (Gesamtes Gehirn)",
    },
    "benzo_bla": {
        "tr": "Yüksek (%60-80 azalma)", "en": "High (60–80% reduction)",
        "de": "Hoch (60–80 % Reduktion)",
    },
    "benzo_blunting": {"tr": "🟡 Orta", "en": "🟡 Moderate", "de": "🟡 Mäßig"},
    "benzo_cognitive": {
        "tr": "🔴 Sedasyon + Bellek kaybı", "en": "🔴 Sedation + Memory impairment",
        "de": "🔴 Sedierung + Gedächtnisstörung",
    },
    "benzo_addiction": {"tr": "🔴 Çok Yüksek", "en": "🔴 Very High", "de": "🔴 Sehr hoch"},
    "benzo_targeting": {
        "tr": "Sistemik (oral/IV)", "en": "Systemic (oral/IV)", "de": "Systemisch (oral/i.v.)",
    },
    "benzo_dosing": {
        "tr": "Kaba (mg cinsinden)", "en": "Coarse (mg scale)", "de": "Grob (mg-Skala)",
    },
    "benzo_reversibility": {
        "tr": "Rebound anksiyete", "en": "Rebound anxiety", "de": "Rebound-Angst",
    },
    "benzo_longterm": {
        "tr": "🔴 Düşük (tolerans)", "en": "🔴 Low (tolerance)", "de": "🔴 Niedrig (Toleranz)",
    },

    # ── Table Ketamine column ──
    "ket_onset": {"tr": "< 2 Saat", "en": "< 2 Hours", "de": "< 2 Stunden"},
    "ket_spatial": {
        "tr": "~%45 (Devre-adaptif)", "en": "~45% (Circuit-adaptive)",
        "de": "~45 % (Schaltkreis-adaptiv)",
    },
    "ket_bla": {
        "tr": "Yüksek (%50-70 azalma)", "en": "High (50–70% reduction)",
        "de": "Hoch (50–70 % Reduktion)",
    },
    "ket_blunting": {"tr": "🟢 Düşük", "en": "🟢 Low", "de": "🟢 Niedrig"},
    "ket_cognitive": {
        "tr": "🟡 Dissosiyasyon (geçici)", "en": "🟡 Dissociation (transient)",
        "de": "🟡 Dissoziation (vorübergehend)",
    },
    "ket_addiction": {"tr": "🟡 Orta", "en": "🟡 Moderate", "de": "🟡 Mäßig"},
    "ket_targeting": {
        "tr": "IV / Nazal spray", "en": "IV / Nasal spray", "de": "i.v. / Nasenspray",
    },
    "ket_dosing": {"tr": "Orta (mg/kg)", "en": "Moderate (mg/kg)", "de": "Mittel (mg/kg)"},
    "ket_reversibility": {
        "tr": "Etki geçici (günler)", "en": "Effect transient (days)",
        "de": "Wirkung vorübergehend (Tage)",
    },
    "ket_longterm": {
        "tr": "🟡 Tekrar doz gerekli", "en": "🟡 Repeat dosing required",
        "de": "🟡 Wiederholungsdosen erforderlich",
    },

    # ── Table IC-LEM column ──
    "iclem_onset": {"tr": "< 100 Milisaniye", "en": "< 100 Milliseconds", "de": "< 100 Millisekunden"},
    "iclem_spatial": {
        "tr": "%98 (Sadece hedef devre)", "en": "98% (Target circuit only)",
        "de": "98 % (Nur Zielschaltkreis)",
    },
    "iclem_blunting": {"tr": "🟢 Yok", "en": "🟢 None", "de": "🟢 Keine"},
    "iclem_cognitive": {"tr": "🟢 Yok", "en": "🟢 None", "de": "🟢 Keine"},
    "iclem_addiction": {"tr": "🟢 Yok", "en": "🟢 None", "de": "🟢 Keine"},
    "iclem_targeting": {
        "tr": "Phased Array (odaklanmış)", "en": "Phased Array (focal)",
        "de": "Phased Array (fokussiert)",
    },
    "iclem_dosing": {
        "tr": "nA hassasiyetinde, gerçek zamanlı",
        "en": "nA precision, real-time adaptive",
        "de": "nA-Präzision, Echtzeit-adaptiv",
    },
    "iclem_reversibility": {
        "tr": "Anında (kapatılabilir)", "en": "Immediate (switchable)",
        "de": "Sofort (abschaltbar)",
    },
    "iclem_longterm": {
        "tr": "🟢 Yüksek (adaptif)", "en": "🟢 High (adaptive)", "de": "🟢 Hoch (adaptiv)",
    },

    # ── Analysis report tab ──
    "analysis_title": {
        "tr": "📝 Otomatik Bilimsel Analiz Raporu",
        "en": "📝 Automated Scientific Analysis Report",
        "de": "📝 Automatisierter wissenschaftlicher Analysebericht",
    },
    "analysis_drug_detail_title": {
        "tr": "💊 İlaç Mekanizma Detayları",
        "en": "💊 Drug Mechanism Details",
        "de": "💊 Details zum Wirkmechanismus",
    },
    "analysis_mechanism": {"tr": "Mekanizma", "en": "Mechanism", "de": "Mechanismus"},
    "analysis_sim_equiv": {
        "tr": "Simülasyon Karşılığı", "en": "Simulation Equivalent",
        "de": "Simulationsäquivalent",
    },
    "analysis_clinical_delay": {
        "tr": "Klinik Gecikme Nedeni", "en": "Clinical Delay Rationale",
        "de": "Klinische Verzögerungsursache",
    },
    "analysis_weakness": {
        "tr": "Ana Zayıflık", "en": "Primary Limitation", "de": "Hauptschwäche",
    },
    "analysis_iclem_rec": {
        "tr": "MÜDAHALe ÖNERİSİ: IC-LEM",
        "en": "INTERVENTION RECOMMENDATION: IC-LEM",
        "de": "INTERVENTIONSEMPFEHLUNG: IC-LEM",
    },

    # ── Analysis templates ──
    "analysis_ssri": {
        "tr": (
            "**[İLAÇ ANALİZİ: SSRI (Fluoxetine)]**\n\n"
            "**Mekanizma:** Sinaptik serotonin seviyelerinde %40 artış simüle edildi. "
            "Tüm ağdaki sinaptik iletim gücü global olarak %10 arttırıldı.\n\n"
            "**Gözlem:** mPFC üzerindeki kontrol etkisi **3 hafta** (simülasyon: 800 ms) "
            "gecikmeyle başladı. Sigmoid aktivasyon profili kullanıldı.\n\n"
            "**Kritik Bulgu:** İlaç sadece Amygdala'yı hedeflemediği için, ödül devrelerinde "
            "(Nucleus Accumbens) **Dopaminerjik körelme** gözlemlendi. "
            "Ateşleme varyansı %40 düştü.\n\n"
            "**Sonuç:** Hasta semptomatik olarak rahatladı ancak **duygusal küntleşme** "
            "(emotional blunting) ve **bilişsel bulanıklık** (brain fog) riski yüksek."
        ),
        "en": (
            "**[DRUG ANALYSIS: SSRI (Fluoxetine)]**\n\n"
            "**Mechanism:** A 40% increase in synaptic serotonin levels was simulated. "
            "Global synaptic transmission gain was elevated by 10% across the entire network.\n\n"
            "**Observation:** The modulatory effect on mPFC manifested with a **3-week** latency "
            "(simulation: 800 ms). A sigmoid activation profile was employed.\n\n"
            "**Critical Finding:** As the drug does not selectively target the amygdala, "
            "**dopaminergic blunting** was observed in the reward circuitry "
            "(Nucleus Accumbens simulation). Firing rate variance decreased by 40%.\n\n"
            "**Conclusion:** The patient achieved symptomatic relief; however, the risk of "
            "**emotional blunting** and **cognitive fog** remains elevated."
        ),
        "de": (
            "**[ARZNEIMITTELANALYSE: SSRI (Fluoxetin)]**\n\n"
            "**Mechanismus:** Ein Anstieg der synaptischen Serotoninspiegel um 40 % wurde simuliert. "
            "Die globale synaptische Übertragungsstärke wurde netzwerkweit um 10 % erhöht.\n\n"
            "**Beobachtung:** Die modulatorische Wirkung auf den mPFC trat mit einer Latenz "
            "von **3 Wochen** (Simulation: 800 ms) ein. Ein sigmoidales Aktivierungsprofil wurde verwendet.\n\n"
            "**Kritischer Befund:** Da das Medikament die Amygdala nicht selektiv adressiert, "
            "wurde eine **dopaminerge Abstumpfung** im Belohnungsschaltkreis "
            "(Nucleus-Accumbens-Simulation) beobachtet. Die Feuerungsratenvarianz sank um 40 %.\n\n"
            "**Schlussfolgerung:** Der Patient erzielte symptomatische Linderung; das Risiko "
            "einer **emotionalen Abstumpfung** und eines **kognitiven Nebels** bleibt jedoch erhöht."
        ),
    },
    "analysis_benzo": {
        "tr": (
            "**[İLAÇ ANALİZİ: Benzodiazepin (Alprazolam/Xanax)]**\n\n"
            "**Mekanizma:** GABA-A reseptör aktivitesi %50 artırıldı. "
            "Beynin genel inhibitör tonusu güçlendirildi.\n\n"
            "**Gözlem:** Amygdala aktivitesi **anında** düştü (latency ≈ 0). "
            "BLA ateşleme hızı hızla azaldı.\n\n"
            "**Kritik Bulgu:** mPFC (karar verme) ve Hippokampüs (bellek) dahil "
            "**tüm kortikal bölgelerde** işlem hızı düştü. "
            "Bağımlılık riski 2-4 hafta sonra başlar.\n\n"
            "**Sonuç:** Akut panik atak için etkili ancak kronik kullanımda "
            "**tolerans gelişimi**, **bilişsel yavaşlama** ve **rebound anksiyete** riski."
        ),
        "en": (
            "**[DRUG ANALYSIS: Benzodiazepine (Alprazolam/Xanax)]**\n\n"
            "**Mechanism:** GABA-A receptor activity was potentiated by 50%. "
            "Global inhibitory tone across the CNS was augmented.\n\n"
            "**Observation:** Amygdala activity decreased **immediately** (latency ≈ 0). "
            "BLA firing rate dropped rapidly.\n\n"
            "**Critical Finding:** Processing speed declined across **all cortical regions**, "
            "including mPFC (executive function) and hippocampus (memory). "
            "Dependence risk emerges within 2–4 weeks.\n\n"
            "**Conclusion:** Effective for acute panic episodes; however, chronic use carries "
            "risk of **tolerance development**, **cognitive slowing**, and **rebound anxiety**."
        ),
        "de": (
            "**[ARZNEIMITTELANALYSE: Benzodiazepin (Alprazolam/Xanax)]**\n\n"
            "**Mechanismus:** Die GABA-A-Rezeptoraktivität wurde um 50 % potenziert. "
            "Der globale inhibitorische Tonus im ZNS wurde verstärkt.\n\n"
            "**Beobachtung:** Die Amygdala-Aktivität sank **sofort** (Latenz ≈ 0). "
            "Die BLA-Feuerungsrate nahm rasch ab.\n\n"
            "**Kritischer Befund:** Die Verarbeitungsgeschwindigkeit sank in **allen kortikalen Regionen**, "
            "einschließlich mPFC (Exekutivfunktion) und Hippocampus (Gedächtnis). "
            "Das Abhängigkeitsrisiko tritt innerhalb von 2–4 Wochen auf.\n\n"
            "**Schlussfolgerung:** Wirksam bei akuten Panikattacken; chronischer Gebrauch birgt jedoch "
            "das Risiko von **Toleranzentwicklung**, **kognitiver Verlangsamung** und **Rebound-Angst**."
        ),
    },
    "analysis_ketamine": {
        "tr": (
            "**[İLAÇ ANALİZİ: Ketamin (Esketamine/Spravato)]**\n\n"
            "**Mekanizma:** NMDA reseptör blokajı → Glutamat patlaması → "
            "AMPA reseptör upregülasyonu → **hızlı sinaptogenez** tetiklendi.\n\n"
            "**Gözlem:** mPFC→BLA sinaptik ağırlık hızla arttı "
            "(top-down kontrol restorasyonu). Ancak etki **geçici**: sönümlenme başladı.\n\n"
            "**Kritik Bulgu:** Ketamin **devre-spesifik plastisite** sağladı. "
            "mPFC-BLA hattı güçlenirken global yan etki profili düşük kaldı.\n\n"
            "**Sonuç:** Tedaviye dirençli depresyonda **saatler** içinde etki. "
            "Ancak **sürdürülebilirlik** ana zayıflık — tekrarlayan dozlama gerekli."
        ),
        "en": (
            "**[DRUG ANALYSIS: Ketamine (Esketamine/Spravato)]**\n\n"
            "**Mechanism:** NMDA receptor blockade → glutamate surge → "
            "AMPA receptor upregulation → **rapid synaptogenesis** triggered.\n\n"
            "**Observation:** mPFC→BLA synaptic weight increased rapidly "
            "(top-down control restoration). However, the effect is **transient**: decay ensued.\n\n"
            "**Critical Finding:** Ketamine induced **circuit-specific plasticity**. "
            "The mPFC-BLA pathway was strengthened while the global side-effect profile remained low.\n\n"
            "**Conclusion:** Effective within **hours** in treatment-resistant depression. "
            "However, **sustainability** is the primary limitation — repeated dosing required."
        ),
        "de": (
            "**[ARZNEIMITTELANALYSE: Ketamin (Esketamin/Spravato)]**\n\n"
            "**Mechanismus:** NMDA-Rezeptorblockade → Glutamat-Burst → "
            "AMPA-Rezeptor-Hochregulation → **schnelle Synaptogenese** ausgelöst.\n\n"
            "**Beobachtung:** Das synaptische Gewicht mPFC→BLA stieg rasch an "
            "(Wiederherstellung der Top-down-Kontrolle). Die Wirkung ist jedoch **vorübergehend**.\n\n"
            "**Kritischer Befund:** Ketamin induzierte **schaltkreisspezifische Plastizität**. "
            "Der mPFC-BLA-Pfad wurde gestärkt, während das globale Nebenwirkungsprofil gering blieb.\n\n"
            "**Schlussfolgerung:** Innerhalb von **Stunden** wirksam bei therapieresistenter Depression. "
            "**Nachhaltigkeit** ist die Hauptschwäche — Wiederholungsdosen erforderlich."
        ),
    },
    "analysis_iclem": {
        "tr": (
            "**[SİSTEM ANALİZİ: IC-LEM Kapalı Döngü Nöromodülasyon]**\n\n"
            "**Mekanizma:** Phased Array mikro-elektrot dizilimi ile BLA-mPFC hattında "
            "**gerçek zamanlı** sense → decode → stimulate döngüsü.\n\n"
            "**Gözlem:** Patolojik aktivite tespit edildi ve milisaniyeler içinde "
            "odaklanmış stimülasyon uygulandı. Sadece hedef devre modüle edildi.\n\n"
            "**Kritik Avantaj:** Sistemin geri kalanı **doğal homeostazında** bırakıldı. "
            "Ödül devreleri, bellek sistemleri ve motor korteks etkilenmedi.\n\n"
            "**Sonuç:** Yan etki profili: **%0**. Stimülasyon sadece gerektiğinde aktif."
        ),
        "en": (
            "**[SYSTEM ANALYSIS: IC-LEM Closed-Loop Neuromodulation]**\n\n"
            "**Mechanism:** Phased array micro-electrode configuration implementing a "
            "**real-time** sense → decode → stimulate loop on the BLA-mPFC pathway.\n\n"
            "**Observation:** Pathological activity was detected and focal stimulation "
            "was delivered within milliseconds. Only the target circuit was modulated.\n\n"
            "**Key Advantage:** The remainder of the neural architecture was maintained in "
            "**natural homeostasis**. Reward circuitry, memory systems, and motor cortex were unaffected.\n\n"
            "**Conclusion:** Side-effect profile: **0%**. Stimulation active only on demand."
        ),
        "de": (
            "**[SYSTEMANALYSE: IC-LEM Closed-Loop-Neuromodulation]**\n\n"
            "**Mechanismus:** Phased-Array-Mikroelektroden-Konfiguration mit einem "
            "**Echtzeit**-Sense → Decode → Stimulate-Regelkreis auf dem BLA-mPFC-Pfad.\n\n"
            "**Beobachtung:** Pathologische Aktivität wurde detektiert und fokale Stimulation "
            "innerhalb von Millisekunden appliziert. Nur der Zielschaltkreis wurde moduliert.\n\n"
            "**Entscheidender Vorteil:** Die übrige neuronale Architektur verblieb in "
            "**natürlicher Homöostase**. Belohnungsschaltkreise, Gedächtnissysteme und Motorkortex "
            "blieben unbeeinflusst.\n\n"
            "**Schlussfolgerung:** Nebenwirkungsprofil: **0 %**. Stimulation nur bei Bedarf aktiv."
        ),
    },

    # ── IC-LEM recommendation (shown under drug analyses) ──
    "iclem_recommendation": {
        "tr": (
            "İlacın aksine, IC-LEM sadece Amygdala'nın patolojik ritmini algıladığında "
            "devreye girer. Global kimyasal değişikliğe gerek kalmadan **Precision Tuning** sağlar. "
            "Ödül devreleri, bellek sistemleri ve motor korteks **doğal homeostazında** bırakılır."
        ),
        "en": (
            "Unlike pharmacotherapy, IC-LEM activates only upon detection of pathological "
            "amygdalar rhythms. It achieves **Precision Tuning** without global neurochemical alteration. "
            "Reward circuits, memory systems, and motor cortex remain in **natural homeostasis**."
        ),
        "de": (
            "Im Gegensatz zur Pharmakotherapie aktiviert sich IC-LEM nur bei Detektion "
            "pathologischer Amygdala-Rhythmen. Es ermöglicht **Precision Tuning** ohne globale "
            "neurochemische Veränderung. Belohnungsschaltkreise, Gedächtnissysteme und Motorkortex "
            "verbleiben in **natürlicher Homöostase**."
        ),
    },

    # ── Drug detail cards ──
    "detail_ssri": {
        "tr": {
            "Mekanizma": "Sinaptik serotonin geri alımını bloke eder → 5-HT seviyesi artar",
            "Simülasyon Karşılığı": "Global sinaptik gain +%10, onset 800ms sigmoid, blunting %40",
            "Klinik Gecikme Nedeni": "Reseptör down-regulation haftalar sürer",
            "Ana Zayıflık": "Hedef seçiciliği yok — tüm serotonerjik yolaklar etkilenir",
        },
        "en": {
            "Mechanism": "Blocks synaptic serotonin reuptake → elevated 5-HT levels",
            "Simulation Equivalent": "Global synaptic gain +10%, onset 800 ms sigmoid, blunting 40%",
            "Clinical Delay Rationale": "Receptor down-regulation requires weeks",
            "Primary Limitation": "No target selectivity — all serotonergic pathways affected",
        },
        "de": {
            "Mechanismus": "Blockiert synaptische Serotonin-Wiederaufnahme → erhöhte 5-HT-Spiegel",
            "Simulationsäquivalent": "Globaler synaptischer Gain +10 %, Onset 800 ms Sigmoid, Abstumpfung 40 %",
            "Klinische Verzögerungsursache": "Rezeptor-Downregulation benötigt Wochen",
            "Hauptschwäche": "Keine Zielselektivität — alle serotonergen Bahnen betroffen",
        },
    },
    "detail_benzo": {
        "tr": {
            "Mekanizma": "GABA-A reseptör pozitif allosterik modülasyonu → inhibitör tonus artar",
            "Simülasyon Karşılığı": "GABA boost +%50, anında etki, BLA sürücü -%60",
            "Klinik Gecikme Nedeni": "Gecikme yok — dakikalar içinde etkili",
            "Ana Zayıflık": "Tolerans + bağımlılık + bilişsel baskılama + rebound",
        },
        "en": {
            "Mechanism": "GABA-A receptor positive allosteric modulation → enhanced inhibitory tone",
            "Simulation Equivalent": "GABA boost +50%, immediate onset, BLA drive −60%",
            "Clinical Delay Rationale": "No delay — effective within minutes",
            "Primary Limitation": "Tolerance + dependence + cognitive suppression + rebound",
        },
        "de": {
            "Mechanismus": "GABA-A-Rezeptor positiv allosterische Modulation → verstärkter inhibitorischer Tonus",
            "Simulationsäquivalent": "GABA-Boost +50 %, sofortiger Wirkungseintritt, BLA-Antrieb −60 %",
            "Klinische Verzögerungsursache": "Keine Verzögerung — innerhalb von Minuten wirksam",
            "Hauptschwäche": "Toleranz + Abhängigkeit + kognitive Suppression + Rebound",
        },
    },
    "detail_ketamine": {
        "tr": {
            "Mekanizma": "NMDA blokajı → Glutamat patlaması → AMPA upregülasyon → sinaptogenez",
            "Simülasyon Karşılığı": "mPFC→BLA plastisite +%45, hızlı onset, decay 1200ms",
            "Klinik Gecikme Nedeni": "Saatler — sinaptik remodeling gerekli",
            "Ana Zayıflık": "Etki geçici — haftada 1-2 doz tekrarı gerekli",
        },
        "en": {
            "Mechanism": "NMDA blockade → glutamate burst → AMPA upregulation → synaptogenesis",
            "Simulation Equivalent": "mPFC→BLA plasticity +45%, rapid onset, decay 1200 ms",
            "Clinical Delay Rationale": "Hours — synaptic remodeling required",
            "Primary Limitation": "Effect transient — weekly re-dosing required",
        },
        "de": {
            "Mechanismus": "NMDA-Blockade → Glutamat-Burst → AMPA-Hochregulation → Synaptogenese",
            "Simulationsäquivalent": "mPFC→BLA Plastizität +45 %, schneller Onset, Abklingen 1200 ms",
            "Klinische Verzögerungsursache": "Stunden — synaptisches Remodeling erforderlich",
            "Hauptschwäche": "Wirkung vorübergehend — wöchentliche Nachdosierung erforderlich",
        },
    },
    "detail_iclem": {
        "tr": {
            "Mekanizma": "Phased array sense→decode→stimulate kapalı döngü",
            "Simülasyon Karşılığı": "Real-time threshold decoding, mPFC stimülasyon, BLA inhibisyon",
            "Klinik Gecikme Nedeni": "Gecikme yok — milisaniye düzeyinde tepki",
            "Ana Zayıflık": "Cerrahi implantasyon gerekli (invaziv)",
        },
        "en": {
            "Mechanism": "Phased array sense → decode → stimulate closed loop",
            "Simulation Equivalent": "Real-time threshold decoding, mPFC stimulation, BLA inhibition",
            "Clinical Delay Rationale": "No delay — millisecond-level response",
            "Primary Limitation": "Surgical implantation required (invasive)",
        },
        "de": {
            "Mechanismus": "Phased Array Sense → Decode → Stimulate Regelkreis",
            "Simulationsäquivalent": "Echtzeit-Schwellenwertdekodierung, mPFC-Stimulation, BLA-Inhibition",
            "Klinische Verzögerungsursache": "Keine Verzögerung — Reaktion im Millisekundenbereich",
            "Hauptschwäche": "Chirurgische Implantation erforderlich (invasiv)",
        },
    },

    # ── Footer ──
    "footer": {
        "tr": (
            'IC-LEM Simülasyon Platformu | "Beyond the Pill" Araştırma Projesi<br>'
            "<em>Bu simülasyon eğitim ve araştırma amaçlıdır. Klinik karar desteği sağlamaz.</em>"
        ),
        "en": (
            'IC-LEM Simulation Platform | "Beyond the Pill" Research Project<br>'
            "<em>This simulation is intended for educational and research purposes only. "
            "It does not provide clinical decision support.</em>"
        ),
        "de": (
            'IC-LEM Simulationsplattform | Forschungsprojekt „Beyond the Pill"<br>'
            "<em>Diese Simulation dient ausschließlich Bildungs- und Forschungszwecken. "
            "Sie bietet keine klinische Entscheidungsunterstützung.</em>"
        ),
    },
}


def t(key, lang="tr"):
    """Retrieve translation. Falls back to Turkish if key or lang missing."""
    entry = T.get(key, {})
    return entry.get(lang, entry.get("tr", f"[{key}]"))
