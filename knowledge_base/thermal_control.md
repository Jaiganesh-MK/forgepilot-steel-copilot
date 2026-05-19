# Thermal control playbook

A cold or thermally weak furnace can be supported by increasing heat input, but the safest lever depends on permeability. Typical advisory levers are coke support, hot-blast temperature increase, small oxygen enrichment increase, and PCI reduction or hold. Avoid aggressive production-push actions when permeability is poor.

# Cold furnace response

When thermal_state_index is negative, hot_metal_temp_c is below target, silicon is low, and predicted_hot_metal_temp_4h_c is falling, diagnose thermal weakness. If permeability is normal, the operator may consider increasing wind or oxygen with care. If permeability is poor, prefer modest thermal support: coke rate increase, hot blast temperature increase if stove margin exists, small oxygen enrichment increase, and PCI reduction or hold.

# Hot furnace response

When thermal_state_index is strongly positive, hot metal temperature and silicon are high, avoid unnecessary thermal input. Consider holding or reducing oxygen enrichment, hot blast temperature, coke support, or PCI depending on permeability and quality risk.
