// SPDX-License-Identifier: MIT

//! Device assignment for models with offloadable sub-components (e.g. MoE experts).

use candle_core::Device;

/// Bundles the primary inference device with the device MoE expert weights
/// load onto.
///
/// Two adjacent same-typed `&Device` parameters invite an unchecked swap at
/// call sites; bundling them into named fields makes that swap a compile-time
/// impossibility instead of a silent bug.
#[derive(Debug, Clone)]
pub struct DeviceAssignment {
    /// Device for model weights and inference (everything but MoE experts).
    pub main: Device,
    /// Device for MoE expert weights. Same as `main` when expert offloading
    /// is not needed; ignored by models and formats without MoE experts.
    pub expert: Device,
}

impl DeviceAssignment {
    /// All weights, including MoE experts, on the same device.
    pub fn uniform(device: &Device) -> Self {
        Self {
            main: device.clone(),
            expert: device.clone(),
        }
    }
}

impl From<&Device> for DeviceAssignment {
    fn from(device: &Device) -> Self {
        Self::uniform(device)
    }
}

impl From<&DeviceAssignment> for DeviceAssignment {
    fn from(devices: &DeviceAssignment) -> Self {
        devices.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verifies `uniform` assigns the same device to both fields.
    #[test]
    fn uniform_assigns_same_device_to_both_fields() {
        let device = Device::Cpu;
        let assignment = DeviceAssignment::uniform(&device);
        assert!(assignment.main.same_device(&assignment.expert));
    }

    // Verifies `From<&Device>` matches `uniform`, so callers that only have
    // a plain `&Device` can rely on `Into<DeviceAssignment>` conversion.
    #[test]
    fn from_device_matches_uniform() {
        let device = Device::Cpu;
        let assignment: DeviceAssignment = (&device).into();
        assert!(assignment.main.same_device(&assignment.expert));
    }
}
