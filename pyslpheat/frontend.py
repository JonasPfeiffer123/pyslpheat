#!/usr/bin/env python3
"""PyQt6 GUI-Frontend für pyslpheat – Wärmelastprofil-Generator."""

import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QPushButton, QGroupBox, QScrollArea, QSplitter,
    QFileDialog, QListWidget, QMessageBox, QRadioButton,
    QButtonGroup,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

import pyslpheat
from pyslpheat import bdew_calculate, vdi4655_calculate
from pyslpheat import (
    TRY_BAUTZEN_2015, TRY_BAUTZEN_2015_WINTER, TRY_BAUTZEN_2015_SUMMER,
    TRY_BAUTZEN_2045, TRY_BAUTZEN_2045_WINTER, TRY_BAUTZEN_2045_SUMMER,
)
from pyslpheat.bdew import compute_holidays

# ── Konstanten ───────────────────────────────────────────────────────────────

TRY_OPTIONS = {
    "Bautzen 2015": TRY_BAUTZEN_2015,
    "Bautzen 2015 – Winter": TRY_BAUTZEN_2015_WINTER,
    "Bautzen 2015 – Sommer": TRY_BAUTZEN_2015_SUMMER,
    "Bautzen 2045": TRY_BAUTZEN_2045,
    "Bautzen 2045 – Winter": TRY_BAUTZEN_2045_WINTER,
    "Bautzen 2045 – Sommer": TRY_BAUTZEN_2045_SUMMER,
    "Eigene Datei…": None,
}

BDEW_PROFILE_TYPES = {
    "HEF – Einfamilienhaus": "HEF",
    "HMF – Mehrfamilienhaus": "HMF",
    "GKO – Büro/Verwaltung": "GKO",
    "GHA – Handel": "GHA",
    "GMK – Metall/Kfz": "GMK",
    "GBD – Bäckerei/Konditorei": "GBD",
    "GBH – Beherbergung": "GBH",
    "GWA – Wäscherei": "GWA",
    "GGA – Gaststätten": "GGA",
    "GBA – Bäder": "GBA",
    "GGB – Gartenbau": "GGB",
    "GPD – Papier/Druck": "GPD",
    "GMF – Mischnutzung": "GMF",
    "GHD – Handel/Dienstleistungen gesamt": "GHD",
}

BDEW_SUBTYPES = {
    "HEF": ["03", "04", "05", "33", "34"],
    "HMF": ["03", "04", "05", "33", "34"],
    "GKO": ["01", "02", "03", "04", "05", "33", "34"],
    "GHA": ["01", "02", "03", "04", "05", "33", "34"],
    "GMK": ["01", "02", "03", "04", "05", "33", "34"],
    "GBD": ["01", "02", "03", "04", "05", "33", "34"],
    "GBH": ["01", "02", "03", "04", "05", "33", "34"],
    "GWA": ["01", "02", "03", "04", "05", "33", "34"],
    "GGA": ["01", "02", "03", "04", "05", "33", "34"],
    "GBA": ["01", "02", "03", "04", "05", "33", "34"],
    "GGB": ["01", "02", "03", "04", "05", "33", "34"],
    "GPD": ["01", "02", "03", "04", "05", "33", "34"],
    "GMF": ["01", "02", "03", "04", "05", "33", "34"],
    "GHD": ["03", "04", "05", "33", "34"],
}

SUBTYPE_LABELS = {
    "01": "01 – sehr gut gedämmt (Niedrigenergiehaus)",
    "02": "02 – gut gedämmt",
    "03": "03 – mittlerer Standard",
    "04": "04 – gering gedämmt",
    "05": "05 – sehr gering gedämmt (Altbau)",
    "33": "33 – FfE-Methode, mittlerer Standard",
    "34": "34 – FfE-Methode, gering gedämmt",
}

VDI_BUILDING_TYPES = {
    "EFH – Einfamilienhaus": "EFH",
    "MFH – Mehrfamilienhaus": "MFH",
    "B – Büro": "B",
}


# ── Worker-Threads ───────────────────────────────────────────────────────────

class CalcWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, fn, params):
        super().__init__()
        self._fn = fn
        self._params = params

    def run(self):
        try:
            result = self._fn(**self._params)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Gemeinsame Hilfswidgets ──────────────────────────────────────────────────

class TRYSelector(QWidget):
    """Combobox für gebündelte TRY-Dateien + optionaler Dateibrowser."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._combo = QComboBox()
        for label in TRY_OPTIONS:
            self._combo.addItem(label)

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Pfad zur TRY-Datei …")
        self._path_edit.setVisible(False)

        self._browse_btn = QPushButton("…")
        self._browse_btn.setFixedWidth(28)
        self._browse_btn.setVisible(False)
        self._browse_btn.clicked.connect(self._browse)

        self._combo.currentTextChanged.connect(self._on_combo)

        layout.addWidget(self._combo)
        layout.addWidget(self._path_edit, 1)
        layout.addWidget(self._browse_btn)

    def _on_combo(self, text):
        custom = text == "Eigene Datei…"
        self._path_edit.setVisible(custom)
        self._browse_btn.setVisible(custom)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "TRY-Datei öffnen", "", "Alle Dateien (*.*)")
        if path:
            self._path_edit.setText(path)

    def get_path(self) -> str:
        label = self._combo.currentText()
        if label == "Eigene Datei…":
            return self._path_edit.text()
        return TRY_OPTIONS[label]


class OptionalLineEdit(QLineEdit):
    """QLineEdit für optionale float-Eingabe (leer → None)."""

    def __init__(self, placeholder="leer = deaktiviert", parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)

    def get_value(self):
        text = self.text().strip().replace(",", ".")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None


# ── Plot-Widget ──────────────────────────────────────────────────────────────

class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(10, 5), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def _base_plot(self, df, ylabel, title, columns_colors_labels):
        self.fig.clear()
        ax1 = self.fig.add_subplot(111)
        ax2 = ax1.twinx()

        for col, color, label in columns_colors_labels:
            ax1.plot(df.index, df[col], color=color, lw=0.7, label=label)

        ax2.plot(df.index, df["temperature_C"], color="#27ae60", lw=0.5,
                 alpha=0.45, label="Temperatur [°C]")

        ax1.set_xlabel("Zeit")
        ax1.set_ylabel(ylabel)
        ax2.set_ylabel("Temperatur [°C]", color="#27ae60")
        ax2.tick_params(axis="y", labelcolor="#27ae60")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
        ax1.set_title(title)
        self.fig.autofmt_xdate()
        self.canvas.draw()

    def plot_bdew(self, df):
        self._base_plot(
            df,
            ylabel="Wärmebedarf [kWh/h]",
            title="BDEW SigLinDe – Stündliches Wärmelastprofil",
            columns_colors_labels=[
                ("Q_total_kWh",  "#2c3e50", "Gesamt"),
                ("Q_heat_kWh",   "#e74c3c", "Raumwärme"),
                ("Q_dhw_kWh",    "#3498db", "Trinkwarmwasser"),
            ],
        )

    def plot_vdi4655(self, df):
        self._base_plot(
            df,
            ylabel="Bedarf [kWh/15 min]",
            title="VDI 4655 – Lastprofil (15-Minuten-Auflösung)",
            columns_colors_labels=[
                ("Q_total_kWh",       "#2c3e50", "Wärme gesamt"),
                ("Q_heat_kWh",        "#e74c3c", "Raumwärme"),
                ("Q_dhw_kWh",         "#3498db", "Trinkwarmwasser"),
                ("Q_electricity_kWh", "#f39c12", "Strom"),
            ],
        )

    def clear(self):
        self.fig.clear()
        self.canvas.draw()


# ── BDEW-Tab ─────────────────────────────────────────────────────────────────

class BDEWTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._result = None
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)

        # ── linke Seite: Parameterleiste ────────────────────────────────────
        left_inner = QWidget()
        left_layout = QVBoxLayout(left_inner)
        left_layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(420)
        scroll.setWidget(left_inner)

        # Grundparameter
        grp = QGroupBox("Grundparameter")
        form = QFormLayout(grp)

        self.profile_combo = QComboBox()
        for lbl in BDEW_PROFILE_TYPES:
            self.profile_combo.addItem(lbl)
        self.profile_combo.currentTextChanged.connect(self._refresh_subtypes)
        form.addRow("Profiltyp:", self.profile_combo)

        self.subtype_combo = QComboBox()
        form.addRow("Subtyp:", self.subtype_combo)

        self.year_spin = QSpinBox()
        self.year_spin.setRange(2000, 2100)
        self.year_spin.setValue(2015)
        form.addRow("Jahr:", self.year_spin)

        self.try_sel = TRYSelector()
        form.addRow("TRY-Datei:", self.try_sel)

        left_layout.addWidget(grp)
        self._refresh_subtypes()

        # Skalierung
        grp2 = QGroupBox("Skalierung")
        vbox2 = QVBoxLayout(grp2)

        mode_row = QHBoxLayout()
        self._mode_grp = QButtonGroup(self)
        self.rb_annual = QRadioButton("Jahresenergie")
        self.rb_design = QRadioButton("Auslegungslast")
        self.rb_both   = QRadioButton("Beides")
        self.rb_annual.setChecked(True)
        for rb in (self.rb_annual, self.rb_design, self.rb_both):
            self._mode_grp.addButton(rb)
            mode_row.addWidget(rb)
        vbox2.addLayout(mode_row)

        scale_form = QFormLayout()

        self.annual_heat_spin = QDoubleSpinBox()
        self.annual_heat_spin.setRange(0, 1e9)
        self.annual_heat_spin.setValue(20000)
        self.annual_heat_spin.setSuffix(" kWh/a")
        self.annual_heat_spin.setDecimals(0)
        scale_form.addRow("Jahreswärmebedarf:", self.annual_heat_spin)

        self.peak_kw_edit = OptionalLineEdit("leer = nicht genutzt")
        scale_form.addRow("Auslegungsleistung [kW]:", self.peak_kw_edit)

        self.design_temp_edit = OptionalLineEdit("leer = nicht genutzt")
        scale_form.addRow("Auslegungstemperatur [°C]:", self.design_temp_edit)

        vbox2.addLayout(scale_form)
        left_layout.addWidget(grp2)

        for rb in (self.rb_annual, self.rb_design, self.rb_both):
            rb.toggled.connect(self._refresh_scaling)
        self._refresh_scaling()

        # Wärmeverteilung
        grp3 = QGroupBox("Wärmeverteilung")
        form3 = QFormLayout(grp3)

        self.dhw_share_edit = OptionalLineEdit("leer = automatisch")
        form3.addRow("TWW-Anteil [0–1]:", self.dhw_share_edit)

        self.heating_limit_edit = OptionalLineEdit("leer = kein Limit")
        form3.addRow("Heizgrenztemperatur [°C]:", self.heating_limit_edit)

        self.heating_exp_spin = QDoubleSpinBox()
        self.heating_exp_spin.setRange(0.1, 5.0)
        self.heating_exp_spin.setValue(1.0)
        self.heating_exp_spin.setSingleStep(0.1)
        self.heating_exp_spin.setDecimals(2)
        form3.addRow("Heizexponent:", self.heating_exp_spin)

        self.dhw_flat_cb = QCheckBox("TWW gleichmäßig verteilen")
        form3.addRow("", self.dhw_flat_cb)

        left_layout.addWidget(grp3)

        # Stochastik
        grp4 = QGroupBox("Stochastik")
        vbox4 = QVBoxLayout(grp4)

        self.stoch_cb = QCheckBox("Stochastische Nachbearbeitung aktivieren")
        vbox4.addWidget(self.stoch_cb)

        self._stoch_params = QWidget()
        stoch_form = QFormLayout(self._stoch_params)
        stoch_form.setContentsMargins(0, 0, 0, 0)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 999999)
        self.seed_spin.setValue(42)
        self.seed_spin.setSpecialValueText("zufällig (None)")
        stoch_form.addRow("Seed:", self.seed_spin)

        self.sigma_sh_spin = QDoubleSpinBox()
        self.sigma_sh_spin.setRange(0.0, 2.0)
        self.sigma_sh_spin.setValue(0.12)
        self.sigma_sh_spin.setSingleStep(0.01)
        self.sigma_sh_spin.setDecimals(3)
        stoch_form.addRow("σ Raumwärme (Log-Normal):", self.sigma_sh_spin)

        self.sigma_dhw_spin = QDoubleSpinBox()
        self.sigma_dhw_spin.setRange(0.0, 2.0)
        self.sigma_dhw_spin.setValue(0.20)
        self.sigma_dhw_spin.setSingleStep(0.01)
        self.sigma_dhw_spin.setDecimals(3)
        stoch_form.addRow("σ TWW (Log-Normal):", self.sigma_dhw_spin)

        self.shift_sh_spin = QSpinBox()
        self.shift_sh_spin.setRange(0, 12)
        self.shift_sh_spin.setValue(1)
        stoch_form.addRow("Max. Peak-Verschiebung Raumwärme [h]:", self.shift_sh_spin)

        self.shift_dhw_spin = QSpinBox()
        self.shift_dhw_spin.setRange(0, 12)
        self.shift_dhw_spin.setValue(2)
        stoch_form.addRow("Max. Peak-Verschiebung TWW [h]:", self.shift_dhw_spin)

        vbox4.addWidget(self._stoch_params)
        left_layout.addWidget(grp4)

        self._stoch_params.setEnabled(False)
        self.stoch_cb.toggled.connect(self._stoch_params.setEnabled)

        # Buttons & Status
        self.run_btn = QPushButton("Berechnen")
        self.run_btn.setFixedHeight(34)
        f = self.run_btn.font(); f.setBold(True); self.run_btn.setFont(f)
        self.run_btn.clicked.connect(self._run)
        left_layout.addWidget(self.run_btn)

        self.export_btn = QPushButton("Ergebnis als CSV exportieren")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export)
        left_layout.addWidget(self.export_btn)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        left_layout.addWidget(self.status_lbl)
        left_layout.addStretch()

        # ── rechte Seite: Plot ───────────────────────────────────────────────
        self.plot = PlotWidget()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(self.plot)
        splitter.setSizes([400, 900])

        root.addWidget(splitter)

    def _refresh_subtypes(self):
        label = self.profile_combo.currentText()
        code = BDEW_PROFILE_TYPES.get(label, "GKO")
        self.subtype_combo.clear()
        for st in BDEW_SUBTYPES.get(code, ["03"]):
            self.subtype_combo.addItem(SUBTYPE_LABELS.get(st, st), userData=st)

    def _refresh_scaling(self):
        annual = self.rb_annual.isChecked() or self.rb_both.isChecked()
        design = self.rb_design.isChecked() or self.rb_both.isChecked()
        self.annual_heat_spin.setEnabled(annual)
        self.peak_kw_edit.setEnabled(design)
        self.design_temp_edit.setEnabled(design)

    def _build_params(self):
        profile_type = BDEW_PROFILE_TYPES[self.profile_combo.currentText()]
        subtype = self.subtype_combo.currentData()
        annual = self.rb_annual.isChecked() or self.rb_both.isChecked()
        design = self.rb_design.isChecked() or self.rb_both.isChecked()
        seed_val = self.seed_spin.value()
        return {
            "annual_heat_kWh":      self.annual_heat_spin.value() if annual else None,
            "profile_type":         profile_type,
            "subtype":              subtype,
            "TRY_file_path":        self.try_sel.get_path(),
            "year":                 self.year_spin.value(),
            "dhw_share":            self.dhw_share_edit.get_value(),
            "heating_limit_temp":   self.heating_limit_edit.get_value(),
            "heating_exponent":     self.heating_exp_spin.value(),
            "dhw_flat":             self.dhw_flat_cb.isChecked(),
            "peak_design_kW":       self.peak_kw_edit.get_value() if design else None,
            "design_temperature":   self.design_temp_edit.get_value() if design else None,
            "stochastic":           self.stoch_cb.isChecked(),
            "stochastic_seed":      seed_val if seed_val >= 0 else None,
            "stochastic_sigma_sh":  self.sigma_sh_spin.value(),
            "stochastic_sigma_dhw": self.sigma_dhw_spin.value(),
            "stochastic_max_shift_sh":  self.shift_sh_spin.value(),
            "stochastic_max_shift_dhw": self.shift_dhw_spin.value(),
        }

    def _run(self):
        if self._worker and self._worker.isRunning():
            return
        self.run_btn.setEnabled(False)
        self.status_lbl.setText("Berechnung läuft …")
        self.plot.clear()

        self._worker = CalcWorker(bdew_calculate, self._build_params())
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, df):
        self._result = df
        self.run_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        total = df["Q_total_kWh"].sum()
        self.status_lbl.setText(
            f"Fertig.  Jahreswärmebedarf: {total:,.0f} kWh/a  |  "
            f"{len(df)} Stundenwerte"
        )
        self.plot.plot_bdew(df)

    def _on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.status_lbl.setText(f"Fehler: {msg}")
        QMessageBox.critical(self, "Berechnungsfehler", msg)

    def _export(self):
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "CSV speichern", "bdew_ergebnis.csv", "CSV-Dateien (*.csv)"
        )
        if path:
            self._result.to_csv(path, sep=";", decimal=",")
            self.status_lbl.setText(f"Exportiert: {path}")


# ── VDI 4655-Tab ─────────────────────────────────────────────────────────────

class VDI4655Tab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._result = None
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)

        left_inner = QWidget()
        left_layout = QVBoxLayout(left_inner)
        left_layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(420)
        scroll.setWidget(left_inner)

        # Grundparameter
        grp = QGroupBox("Grundparameter")
        form = QFormLayout(grp)

        self.building_combo = QComboBox()
        for lbl in VDI_BUILDING_TYPES:
            self.building_combo.addItem(lbl)
        form.addRow("Gebäudetyp:", self.building_combo)

        self.persons_spin = QSpinBox()
        self.persons_spin.setRange(1, 20)
        self.persons_spin.setValue(3)
        form.addRow("Personen im Haushalt:", self.persons_spin)

        self.year_spin = QSpinBox()
        self.year_spin.setRange(2000, 2100)
        self.year_spin.setValue(2019)
        form.addRow("Jahr:", self.year_spin)

        self.climate_combo = QComboBox()
        for z in [str(i) for i in range(1, 16)]:
            self.climate_combo.addItem(z)
        self.climate_combo.setCurrentText("9")
        form.addRow("Klimazone (1–15):", self.climate_combo)

        self.try_sel = TRYSelector()
        form.addRow("TRY-Datei:", self.try_sel)

        left_layout.addWidget(grp)

        # Jahresenergien
        grp2 = QGroupBox("Jahresenergien")
        form2 = QFormLayout(grp2)

        self.heating_spin = QDoubleSpinBox()
        self.heating_spin.setRange(0, 1e9)
        self.heating_spin.setValue(12000)
        self.heating_spin.setSuffix(" kWh/a")
        self.heating_spin.setDecimals(0)
        form2.addRow("Raumwärme:", self.heating_spin)

        self.dhw_spin = QDoubleSpinBox()
        self.dhw_spin.setRange(0, 1e9)
        self.dhw_spin.setValue(2000)
        self.dhw_spin.setSuffix(" kWh/a")
        self.dhw_spin.setDecimals(0)
        form2.addRow("Trinkwarmwasser:", self.dhw_spin)

        self.elec_spin = QDoubleSpinBox()
        self.elec_spin.setRange(0, 1e9)
        self.elec_spin.setValue(3500)
        self.elec_spin.setSuffix(" kWh/a")
        self.elec_spin.setDecimals(0)
        form2.addRow("Strom:", self.elec_spin)

        left_layout.addWidget(grp2)

        # Feiertage
        grp3 = QGroupBox("Feiertage")
        vbox3 = QVBoxLayout(grp3)

        self.holiday_list = QListWidget()
        self.holiday_list.setMaximumHeight(130)
        self.holiday_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        vbox3.addWidget(self.holiday_list)

        date_row = QHBoxLayout()
        self.holiday_edit = QLineEdit()
        self.holiday_edit.setPlaceholderText("YYYY-MM-DD eingeben …")
        self.holiday_edit.returnPressed.connect(self._add_holiday)
        add_btn = QPushButton("Hinzufügen")
        add_btn.clicked.connect(self._add_holiday)
        rem_btn = QPushButton("Entfernen")
        rem_btn.clicked.connect(self._remove_holidays)
        date_row.addWidget(self.holiday_edit)
        date_row.addWidget(add_btn)
        date_row.addWidget(rem_btn)
        vbox3.addLayout(date_row)

        auto_btn = QPushButton("Gesetzliche Feiertage (DE) für gewähltes Jahr einfügen")
        auto_btn.clicked.connect(self._insert_default_holidays)
        vbox3.addWidget(auto_btn)

        left_layout.addWidget(grp3)

        # Buttons & Status
        self.run_btn = QPushButton("Berechnen")
        self.run_btn.setFixedHeight(34)
        f = self.run_btn.font(); f.setBold(True); self.run_btn.setFont(f)
        self.run_btn.clicked.connect(self._run)
        left_layout.addWidget(self.run_btn)

        self.export_btn = QPushButton("Ergebnis als CSV exportieren")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export)
        left_layout.addWidget(self.export_btn)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        left_layout.addWidget(self.status_lbl)
        left_layout.addStretch()

        # rechte Seite
        self.plot = PlotWidget()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(self.plot)
        splitter.setSizes([400, 900])

        root.addWidget(splitter)

    # ── Feiertage-Helfer ────────────────────────────────────────────────────

    def _existing_dates(self):
        return {self.holiday_list.item(i).text() for i in range(self.holiday_list.count())}

    def _add_holiday(self):
        text = self.holiday_edit.text().strip()
        if not text:
            return
        try:
            np.datetime64(text, "D")  # Validierung
        except Exception:
            QMessageBox.warning(self, "Ungültiges Datum",
                                f"'{text}' ist kein gültiges Datum (Format: YYYY-MM-DD).")
            return
        if text not in self._existing_dates():
            self.holiday_list.addItem(text)
        self.holiday_edit.clear()

    def _remove_holidays(self):
        for item in self.holiday_list.selectedItems():
            self.holiday_list.takeItem(self.holiday_list.row(item))

    def _insert_default_holidays(self):
        year = self.year_spin.value()
        existing = self._existing_dates()
        for h in sorted(compute_holidays(year)):
            date_str = str(h)
            if date_str not in existing:
                self.holiday_list.addItem(date_str)
                existing.add(date_str)

    def _get_holidays(self):
        items = [self.holiday_list.item(i).text() for i in range(self.holiday_list.count())]
        if not items:
            return np.array([], dtype="datetime64[D]")
        return np.array(items, dtype="datetime64[D]")

    # ── Berechnung ──────────────────────────────────────────────────────────

    def _build_params(self):
        building_type = VDI_BUILDING_TYPES[self.building_combo.currentText()]
        return {
            "annual_heating_kWh":     self.heating_spin.value(),
            "annual_dhw_kWh":         self.dhw_spin.value(),
            "annual_electricity_kWh": self.elec_spin.value(),
            "building_type":          building_type,
            "number_people_household": self.persons_spin.value(),
            "year":                   self.year_spin.value(),
            "climate_zone":           self.climate_combo.currentText(),
            "TRY":                    self.try_sel.get_path(),
            "holidays":               self._get_holidays(),
        }

    def _run(self):
        if self._worker and self._worker.isRunning():
            return
        self.run_btn.setEnabled(False)
        self.status_lbl.setText("Berechnung läuft …")
        self.plot.clear()

        self._worker = CalcWorker(vdi4655_calculate, self._build_params())
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, df):
        self._result = df
        self.run_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        heat = df["Q_total_kWh"].sum()
        elec = df["Q_electricity_kWh"].sum()
        self.status_lbl.setText(
            f"Fertig.  Wärme: {heat:,.0f} kWh/a  |  Strom: {elec:,.0f} kWh/a  |  "
            f"{len(df)} Viertelstundenwerte"
        )
        self.plot.plot_vdi4655(df)

    def _on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.status_lbl.setText(f"Fehler: {msg}")
        QMessageBox.critical(self, "Berechnungsfehler", msg)

    def _export(self):
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "CSV speichern", "vdi4655_ergebnis.csv", "CSV-Dateien (*.csv)"
        )
        if path:
            self._result.to_csv(path, sep=";", decimal=",")
            self.status_lbl.setText(f"Exportiert: {path}")


# ── Hauptfenster ─────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        version = getattr(pyslpheat, "__version__", "?")
        self.setWindowTitle(f"pyslpheat {version} – Wärmelastprofil-Generator")
        self.setMinimumSize(1100, 680)
        self.resize(1350, 800)

        tabs = QTabWidget()
        tabs.addTab(BDEWTab(), "BDEW SigLinDe (stündlich)")
        tabs.addTab(VDI4655Tab(), "VDI 4655 (15-Minuten)")
        self.setCentralWidget(tabs)

        self.statusBar().showMessage(
            f"pyslpheat v{version}"
        )


# ── Einstiegspunkt ───────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("pyslpheat")
    app.setOrganizationName("HSZG Energiespeicher")
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
