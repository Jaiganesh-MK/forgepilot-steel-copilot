from __future__ import annotations

from typing import Any

from .agent_core import BaseAgent, PlantContext, AgentSignal, clamp, safe_float, safe_str


def trend(history, column: str, periods: int = 6) -> float:
    if history is None or history.empty or column not in history.columns:
        return 0.0
    series = history[column].dropna().tail(periods)
    if len(series) < 2:
        return 0.0
    return safe_float(series.iloc[-1]) - safe_float(series.iloc[0])


def ev(label: str, value: Any, unit: str = "") -> str:
    suffix = f" {unit}" if unit else ""
    return f"{label}: {value}{suffix}"


class ThermalStateAgent(BaseAgent):
    name = "ThermalStateAgent"
    decision_area = "Thermal balance"

    def evaluate(self, context: PlantContext) -> AgentSignal:
        row = context.current
        thermal = safe_float(row.get("thermal_state_index"))
        temp = safe_float(row.get("hot_metal_temp_c"))
        pred_temp = safe_float(row.get("predicted_hot_metal_temp_4h_c"))
        si = safe_float(row.get("hot_metal_si_pct"))
        pred_si = safe_float(row.get("predicted_si_4h_pct"))
        risk = safe_float(row.get("thermal_risk_score"))
        evidence = [
            ev("Thermal state index", round(thermal, 2)), ev("Hot metal temp", round(temp, 1), "C"),
            ev("Predicted 4h temp", round(pred_temp, 1), "C"), ev("Hot metal Si", round(si, 3), "%"),
            ev("Predicted 4h Si", round(pred_si, 3), "%"), ev("6h temp trend", round(trend(context.history, "hot_metal_temp_c"), 1), "C"),
        ]
        if thermal < -0.85 or pred_temp < 1475 or pred_si < 0.36:
            return AgentSignal(self.name, self.decision_area, "critical" if risk >= 70 else "high", clamp(0.68 + risk / 250, 0.7, 0.94), "Furnace is drifting cold; restore heat input without pushing gas flow aggressively.", evidence, {"coke_rate_delta_kg_thm": 10, "pci_delta_kg_thm": -8, "oxygen_enrichment_delta_pct": 0.2, "blast_temp_delta_c": 10}, ["CokeRateAgent", "PCIAgent", "OxygenEnrichmentAgent", "BlastTemperatureAgent"], ["Confirm lab Si delay and sensor quality before acting.", "Check burden moisture and coke CSR trend."], ["cold_drift", "quality_variability"])
        if thermal > 0.85 or pred_temp > 1518 or pred_si > 0.68:
            return AgentSignal(self.name, self.decision_area, "critical" if risk >= 70 else "high", clamp(0.66 + risk / 260, 0.68, 0.92), "Furnace is hot; reduce excess heat input using small trims and watch Si response.", evidence, {"coke_rate_delta_kg_thm": -8, "pci_delta_kg_thm": 6, "oxygen_enrichment_delta_pct": -0.2, "blast_temp_delta_c": -5}, ["CokeRateAgent", "PCIAgent", "OxygenEnrichmentAgent"], ["Verify trend is not caused by delayed sample timing.", "Avoid simultaneous large fuel and oxygen changes."], ["hot_drift", "silicon_variability"])
        return AgentSignal(self.name, self.decision_area, "low", 0.74, "Thermal state is within the advisory band; maintain fuel balance and monitor trend.", evidence, {}, [], [], ["stable_thermal_state"])


class PermeabilityAgent(BaseAgent):
    name = "PermeabilityAgent"
    decision_area = "Gas flow and permeability"

    def evaluate(self, context: PlantContext) -> AgentSignal:
        row = context.current
        pressure_drop = safe_float(row.get("pressure_drop_kpa"))
        perm = safe_float(row.get("permeability_index"))
        tuyere_blocks = safe_float(row.get("tuyere_blockage_count"))
        risk = safe_float(row.get("permeability_risk_score"))
        severe = perm < 70 or pressure_drop > 195 or tuyere_blocks >= 2
        evidence = [
            ev("Pressure drop", round(pressure_drop, 1), "kPa"), ev("Permeability index", round(perm, 1)),
            ev("6h pressure-drop trend", round(trend(context.history, "pressure_drop_kpa"), 1), "kPa"),
            ev("6h permeability trend", round(trend(context.history, "permeability_index"), 1)),
            ev("Tuyere blockage count", int(tuyere_blocks)),
        ]
        if perm < 80 or pressure_drop > 180 or tuyere_blocks >= 1:
            return AgentSignal(self.name, self.decision_area, "critical" if severe else "high", clamp(0.70 + risk / 230, 0.72, 0.95), "Permeability stress detected; stabilize gas flow before increasing production intensity.", evidence, {"wind_volume_delta_nm3_min": -120 if severe else -70, "pci_delta_kg_thm": -10 if severe else -6, "coke_rate_delta_kg_thm": 8 if severe else 5, "top_pressure_delta_kpa": 5, "burden_distribution_change": "Move one step toward center-coke / flow-recovery charging pattern"}, ["WindVolumeAgent", "PCIAgent", "CokeRateAgent", "TopPressureAgent", "BurdenDistributionAgent"], ["Check tuyere camera or thermal image for blockage.", "Confirm no instrument drift on pressure taps."], ["permeability_loss", "slip_or_hang_risk"])
        return AgentSignal(self.name, self.decision_area, "low", 0.77, "Gas flow and permeability are not showing recovery-risk conditions.", evidence, {}, [], [], ["stable_permeability"])


class WindVolumeAgent(BaseAgent):
    name = "WindVolumeAgent"
    decision_area = "Wind volume"

    def evaluate(self, context: PlantContext) -> AgentSignal:
        row = context.current
        wind = safe_float(row.get("wind_volume_nm3_min"))
        perm = safe_float(row.get("permeability_index"))
        pressure_drop = safe_float(row.get("pressure_drop_kpa"))
        thermal = safe_float(row.get("thermal_state_index"))
        risk = max(safe_float(row.get("permeability_risk_score")), safe_float(row.get("thermal_risk_score")))
        evidence = [ev("Wind volume", round(wind, 0), "Nm3/min"), ev("Pressure drop", round(pressure_drop, 1), "kPa"), ev("Permeability index", round(perm, 1)), ev("Thermal state index", round(thermal, 2))]
        if perm < 80 or pressure_drop > 180:
            delta = -120 if perm < 70 or pressure_drop > 195 else -70
            return AgentSignal(self.name, self.decision_area, "high", clamp(0.70 + risk / 260, 0.72, 0.93), "Reduce wind volume temporarily to relieve permeability stress.", evidence, {"wind_volume_delta_nm3_min": delta}, ["PermeabilityAgent", "TopPressureAgent", "BurdenDistributionAgent"], ["Coordinate with PCI and top-pressure changes; avoid isolated wind changes."], ["wind_reduction", "permeability_recovery"])
        if thermal > 0.9 and perm > 86 and pressure_drop < 170:
            return AgentSignal(self.name, self.decision_area, "medium", 0.70, "Permeability allows a modest wind-volume increase to recover productivity and moderate a hot trend.", evidence, {"wind_volume_delta_nm3_min": 50}, ["ThermalStateAgent"], ["Confirm no recent slips or unstable descent before increasing wind."], ["productivity_trim", "hot_state_moderation"])
        if thermal < -0.9:
            return AgentSignal(self.name, self.decision_area, "medium", 0.67, "Avoid increasing wind while the furnace is cold; a small reduction may help restore thermal balance.", evidence, {"wind_volume_delta_nm3_min": -40}, ["ThermalStateAgent", "CokeRateAgent"], ["Pair with fuel-side correction; do not treat wind as the only correction variable."], ["cold_state_caution"])
        return AgentSignal(self.name, self.decision_area, "low", 0.75, "Maintain current wind volume.", evidence, {}, [], [], ["maintain_wind"])


class PCIAgent(BaseAgent):
    name = "PCIAgent"
    decision_area = "PCI injection rate"

    def evaluate(self, context: PlantContext) -> AgentSignal:
        row = context.current
        pci = safe_float(row.get("pci_rate_kg_thm"))
        raft = safe_float(row.get("raceway_adiabatic_flame_temp_c"))
        gas_use = safe_float(row.get("gas_utilization_pct"))
        thermal = safe_float(row.get("thermal_state_index"))
        perm = safe_float(row.get("permeability_index"))
        evidence = [ev("PCI rate", round(pci, 1), "kg/thm"), ev("RAFT", round(raft, 0), "C"), ev("Gas utilization", round(gas_use, 1), "%"), ev("Thermal state index", round(thermal, 2)), ev("Permeability index", round(perm, 1))]
        if thermal < -0.75 or raft < 2075 or perm < 80:
            return AgentSignal(self.name, self.decision_area, "high" if perm < 80 or thermal < -0.95 else "medium", 0.79 if perm < 80 else 0.73, "Reduce PCI temporarily; current state needs coke support and heat/permeability recovery before high coal replacement.", evidence, {"pci_delta_kg_thm": -12 if thermal < -1.0 or perm < 72 else -8}, ["CokeRateAgent", "OxygenEnrichmentAgent", "ThermalStateAgent"], ["Confirm coal injection system stability and tuyere observations."], ["pci_reduction", "heat_recovery"])
        if thermal > 0.8 and perm > 86 and gas_use >= 47.0:
            return AgentSignal(self.name, self.decision_area, "medium", 0.72, "A small PCI increase can substitute coke and moderate a hot trend if permeability remains healthy.", evidence, {"pci_delta_kg_thm": 6}, ["CokeRateAgent", "OxygenEnrichmentAgent"], ["Check that RAFT margin remains acceptable after the increase."], ["coke_substitution", "hot_state_moderation"])
        return AgentSignal(self.name, self.decision_area, "low", 0.74, "Maintain PCI rate.", evidence, {}, [], [], ["maintain_pci"])


class CokeRateAgent(BaseAgent):
    name = "CokeRateAgent"
    decision_area = "Coke rate"

    def evaluate(self, context: PlantContext) -> AgentSignal:
        row = context.current
        coke_rate = safe_float(row.get("coke_rate_kg_thm"))
        thermal = safe_float(row.get("thermal_state_index"))
        csr = safe_float(row.get("coke_csr_pct"))
        ash = safe_float(row.get("coke_ash_pct"))
        m10 = safe_float(row.get("coke_m10_pct"))
        perm = safe_float(row.get("permeability_index"))
        weak_coke = csr < 61 or ash > 12.8 or m10 > 8.2
        evidence = [ev("Coke rate", round(coke_rate, 1), "kg/thm"), ev("Thermal state index", round(thermal, 2)), ev("Coke CSR", round(csr, 1), "%"), ev("Coke ash", round(ash, 2), "%"), ev("Coke M10", round(m10, 2), "%"), ev("Permeability index", round(perm, 1))]
        if thermal < -0.7 or weak_coke or perm < 80:
            return AgentSignal(self.name, self.decision_area, "high" if thermal < -0.9 or perm < 80 or weak_coke else "medium", 0.80 if weak_coke else 0.74, "Increase coke support to recover heat/permeability margin; pair with PCI trim if needed.", evidence, {"coke_rate_delta_kg_thm": 12 if thermal < -1.0 or weak_coke else 6}, ["PCIAgent", "ThermalStateAgent", "PermeabilityAgent"], ["Validate coke-quality result and burden mix before changing base rate."], ["coke_support", "thermal_or_permeability_recovery"])
        if thermal > 0.85 and perm > 84:
            return AgentSignal(self.name, self.decision_area, "medium", 0.70, "Reduce coke rate slightly; thermal state is hot and permeability allows a controlled fuel trim.", evidence, {"coke_rate_delta_kg_thm": -8}, ["PCIAgent", "ThermalStateAgent"], ["Avoid reducing coke during incipient permeability loss."], ["fuel_trim", "hot_state_moderation"])
        return AgentSignal(self.name, self.decision_area, "low", 0.74, "Maintain coke rate.", evidence, {}, [], [], ["maintain_coke"])


class FuelRateAgent(BaseAgent):
    name = "FuelRateAgent"
    decision_area = "Total fuel rate"

    def evaluate(self, context: PlantContext) -> AgentSignal:
        row = context.current
        fuel = safe_float(row.get("total_fuel_rate_kg_thm"))
        target = safe_float(row.get("fuel_rate_target_kg_thm"))
        thermal = safe_float(row.get("thermal_state_index"))
        deviation = fuel - target
        evidence = [ev("Total fuel rate", round(fuel, 1), "kg/thm"), ev("Fuel target", round(target, 1), "kg/thm"), ev("Deviation", round(deviation, 1), "kg/thm"), ev("Thermal state index", round(thermal, 2))]
        if abs(deviation) > 18 and abs(thermal) > 0.7:
            action = "Rebalance coke/PCI mix rather than changing only total fuel."
            return AgentSignal(self.name, self.decision_area, "medium", 0.66, action, evidence, {"monitoring_action": "Flag fuel-rate deviation and require coke/PCI cross-check"}, ["PCIAgent", "CokeRateAgent", "ThermalStateAgent"], ["Check if deviation is intentional due to raw material change."], ["fuel_rate_deviation"])
        return AgentSignal(self.name, self.decision_area, "low", 0.72, "Total fuel rate is not driving an active advisory.", evidence, {}, [], [], ["maintain_fuel_rate"])


class OxygenEnrichmentAgent(BaseAgent):
    name = "OxygenEnrichmentAgent"
    decision_area = "Oxygen enrichment"

    def evaluate(self, context: PlantContext) -> AgentSignal:
        row = context.current
        oxygen = safe_float(row.get("oxygen_enrichment_pct"))
        raft = safe_float(row.get("raceway_adiabatic_flame_temp_c"))
        thermal = safe_float(row.get("thermal_state_index"))
        perm = safe_float(row.get("permeability_index"))
        evidence = [ev("Oxygen enrichment", round(oxygen, 2), "%"), ev("RAFT", round(raft, 0), "C"), ev("Thermal state index", round(thermal, 2)), ev("Permeability index", round(perm, 1))]
        if thermal < -0.75 and perm >= 76 and raft < 2145:
            return AgentSignal(self.name, self.decision_area, "medium", 0.71, "Apply a small oxygen-enrichment increase to support heat recovery, coordinated with PCI and coke actions.", evidence, {"oxygen_enrichment_delta_pct": 0.2}, ["PCIAgent", "CokeRateAgent", "ThermalStateAgent"], ["Check oxygen plant availability and tuyere limits."], ["heat_recovery", "oxygen_trim"])
        if thermal > 0.85 or raft > 2185:
            return AgentSignal(self.name, self.decision_area, "medium", 0.69, "Reduce oxygen enrichment slightly to avoid reinforcing a hot trend.", evidence, {"oxygen_enrichment_delta_pct": -0.2}, ["ThermalStateAgent"], ["Coordinate with wind and fuel-rate trims to avoid productivity shock."], ["hot_state_moderation"])
        return AgentSignal(self.name, self.decision_area, "low", 0.73, "Maintain oxygen enrichment.", evidence, {}, [], [], ["maintain_oxygen"])


class BlastTemperatureAgent(BaseAgent):
    name = "BlastTemperatureAgent"
    decision_area = "Hot blast temperature"

    def evaluate(self, context: PlantContext) -> AgentSignal:
        row = context.current
        blast_temp = safe_float(row.get("hot_blast_temp_c"))
        thermal = safe_float(row.get("thermal_state_index"))
        pred_temp = safe_float(row.get("predicted_hot_metal_temp_4h_c"))
        evidence = [ev("Hot blast temperature", round(blast_temp, 0), "C"), ev("Thermal state index", round(thermal, 2)), ev("Predicted 4h hot metal temp", round(pred_temp, 1), "C")]
        if thermal < -0.7 and blast_temp < 1205:
            return AgentSignal(self.name, self.decision_area, "medium", 0.70, "Increase hot blast temperature if stove margin is available; this is a lower-disturbance heat-recovery lever.", evidence, {"blast_temp_delta_c": 10}, ["ThermalStateAgent"], ["Confirm stove availability and maximum blast-temperature limit."], ["heat_recovery"])
        if thermal > 0.95 and blast_temp > 1185:
            return AgentSignal(self.name, self.decision_area, "low", 0.64, "A small blast-temperature reduction may help moderate a hot trend if fuel trims are insufficient.", evidence, {"blast_temp_delta_c": -5}, ["ThermalStateAgent"], ["Use only after confirming persistent hot trend."], ["hot_state_moderation"])
        return AgentSignal(self.name, self.decision_area, "low", 0.72, "Maintain hot blast temperature.", evidence, {}, [], [], ["maintain_blast_temperature"])


class TopPressureAgent(BaseAgent):
    name = "TopPressureAgent"
    decision_area = "Top pressure"

    def evaluate(self, context: PlantContext) -> AgentSignal:
        row = context.current
        top_pressure = safe_float(row.get("top_pressure_kpa"))
        pressure_drop = safe_float(row.get("pressure_drop_kpa"))
        perm = safe_float(row.get("permeability_index"))
        evidence = [ev("Top pressure", round(top_pressure, 1), "kPa"), ev("Pressure drop", round(pressure_drop, 1), "kPa"), ev("Permeability index", round(perm, 1))]
        if perm < 80 or pressure_drop > 185:
            return AgentSignal(self.name, self.decision_area, "medium", 0.66, "Consider a small top-pressure increase to dampen gas-flow instability, if equipment constraints allow.", evidence, {"top_pressure_delta_kpa": 5}, ["PermeabilityAgent", "WindVolumeAgent"], ["Confirm TRT/top-pressure control availability and upper pressure limit."], ["gas_flow_stabilization"])
        return AgentSignal(self.name, self.decision_area, "low", 0.71, "Maintain top pressure.", evidence, {}, [], [], ["maintain_top_pressure"])


class BurdenDistributionAgent(BaseAgent):
    name = "BurdenDistributionAgent"
    decision_area = "Burden distribution"

    def evaluate(self, context: PlantContext) -> AgentSignal:
        row = context.current
        mode = safe_str(row.get("burden_distribution_mode"), "unknown")
        left = safe_float(row.get("stockline_left_m"))
        right = safe_float(row.get("stockline_right_m"))
        asym = abs(left - right)
        perm = safe_float(row.get("permeability_index"))
        pressure_drop = safe_float(row.get("pressure_drop_kpa"))
        evidence = [ev("Burden distribution mode", mode), ev("Stockline asymmetry", round(asym, 2), "m"), ev("Permeability index", round(perm, 1)), ev("Pressure drop", round(pressure_drop, 1), "kPa")]
        if perm < 80 or pressure_drop > 180 or asym > 0.22:
            return AgentSignal(self.name, self.decision_area, "high" if perm < 70 else "medium", 0.68 if asym < 0.22 else 0.76, "Use a flow-recovery burden distribution pattern; avoid aggressive production push until descent stabilizes.", evidence, {"burden_distribution_change": "One-step center-coke / permeability-recovery matrix"}, ["PermeabilityAgent", "WindVolumeAgent"], ["Confirm charging equipment status and stockline measurement quality."], ["burden_distribution", "permeability_recovery"])
        return AgentSignal(self.name, self.decision_area, "low", 0.70, "Maintain current burden distribution program.", evidence, {}, [], [], ["maintain_burden_distribution"])


class TappingAgent(BaseAgent):
    name = "TappingAgent"
    decision_area = "Tapping priority"

    def evaluate(self, context: PlantContext) -> AgentSignal:
        row = context.current
        hearth_level = safe_float(row.get("hearth_liquid_level_index"))
        hearth_temp = safe_float(row.get("hearth_sidewall_temp_c"))
        production = safe_float(row.get("production_tph"))
        evidence = [ev("Hearth liquid level index", round(hearth_level, 1)), ev("Hearth sidewall temp", round(hearth_temp, 1), "C"), ev("Production", round(production, 1), "tph")]
        if hearth_level > 78:
            return AgentSignal(self.name, self.decision_area, "high" if hearth_level > 86 else "medium", 0.78 if hearth_level > 86 else 0.70, "Raise tapping priority to prevent hearth-level-related instability.", evidence, {"tapping_priority": "High - prepare tap and avoid delay"}, ["ThermalStateAgent", "PermeabilityAgent"], ["Confirm cast-house readiness and tap-hole condition."], ["hearth_level", "tapping_priority"])
        return AgentSignal(self.name, self.decision_area, "low", 0.76, "Normal tapping priority.", evidence, {}, [], [], ["normal_tapping"])


class QualityAgent(BaseAgent):
    name = "QualityAgent"
    decision_area = "Hot metal quality"

    def evaluate(self, context: PlantContext) -> AgentSignal:
        row = context.current
        si = safe_float(row.get("hot_metal_si_pct"))
        pred_si = safe_float(row.get("predicted_si_4h_pct"))
        sulfur = safe_float(row.get("hot_metal_s_pct"))
        quality_risk = safe_float(row.get("quality_risk_score"))
        sensor_flag = safe_str(row.get("sensor_quality_flag"), "Unknown")
        evidence = [ev("Hot metal Si", round(si, 3), "%"), ev("Predicted 4h Si", round(pred_si, 3), "%"), ev("Hot metal S", round(sulfur, 3), "%"), ev("Quality risk score", round(quality_risk, 1)), ev("Sensor quality flag", sensor_flag)]
        if si < 0.36 or pred_si < 0.36:
            return AgentSignal(self.name, self.decision_area, "high" if quality_risk >= 60 else "medium", 0.70, "Low silicon risk; validate sample timing and coordinate with thermal recovery actions.", evidence, {"monitoring_action": "Repeat/validate lab sample and watch predicted Si trend"}, ["ThermalStateAgent", "CokeRateAgent", "PCIAgent"], ["Do not chase a single lab point without trend confirmation."], ["low_silicon_risk"])
        if si > 0.68 or pred_si > 0.68:
            return AgentSignal(self.name, self.decision_area, "high" if quality_risk >= 60 else "medium", 0.70, "High silicon risk; use small fuel/oxygen trims and avoid over-correction.", evidence, {"monitoring_action": "Repeat/validate lab sample and watch predicted Si trend"}, ["ThermalStateAgent", "CokeRateAgent", "OxygenEnrichmentAgent"], ["Check whether sample lag explains the apparent excursion."], ["high_silicon_risk"])
        if sensor_flag.lower() != "good":
            return AgentSignal(self.name, self.decision_area, "medium", 0.65, "Sensor or lab quality is not good; validate measurements before changing setpoints.", evidence, {"monitoring_action": "Validate sensor/lab data before changing setpoints"}, ["SafetyGateAgent"], ["Check sensor and lab quality flags."], ["data_quality"])
        return AgentSignal(self.name, self.decision_area, "low", 0.76, "Hot metal quality is within target band.", evidence, {}, [], [], ["quality_in_band"])
