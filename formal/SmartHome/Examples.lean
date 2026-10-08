import SmartHome.Allocation

/-!
# Examples

The lighting card from the extension plan, a thermostat card, and a few batches that the
checker should accept or reject. The `#guard` lines are checked at build time.
-/

namespace SmartHome.Examples

/-- The lighting agent card from the extension plan (booleans encoded as `0`/`1`). -/
def lighting : AgentCard :=
  { device := "lighting"
    ops := [
      { name := "set_brightness"
        params := [{ name := "level", lo := 0, hi := 100 }]
        reads := ["light.online", "light.brightness"]
        writes := ["light.brightness"]
        pre := [.eqConst "light.online" 1]
        post := [.eqParam "light.brightness" "level"] },
      { name := "set_color_temperature"
        params := [{ name := "kelvin", lo := 2000, hi := 6500 }]
        reads := ["light.online"]
        writes := ["light.color_temperature"]
        pre := [.eqConst "light.online" 1]
        post := [.eqParam "light.color_temperature" "kelvin"] }] }

/-- A thermostat agent card in the same style. -/
def thermostat : AgentCard :=
  { device := "thermostat"
    ops := [
      { name := "set_temperature"
        params := [{ name := "celsius", lo := 5, hi := 30 }]
        reads := ["thermostat.online"]
        writes := ["thermostat.target"]
        pre := [.eqConst "thermostat.online" 1]
        post := [.eqParam "thermostat.target" "celsius"] }] }

def cards : List AgentCard := [lighting, thermostat]

/-- Observed state: both devices online. -/
def state : State :=
  [("light.online", 1), ("light.brightness", 70), ("thermostat.online", 1),
    ("thermostat.target", 21)]

/-- Observed state: the lights are offline. -/
def stateLightsOffline : State :=
  [("light.online", 0), ("light.brightness", 70), ("thermostat.online", 1),
    ("thermostat.target", 21)]

/-- Dim the lights and warm the room: two devices, no shared writes. -/
def goodBatch : Batch :=
  [{ device := "lighting", op := "set_brightness", args := [("level", 30)] },
    { device := "thermostat", op := "set_temperature", args := [("celsius", 23)] }]

/-- Two tasks write `light.brightness`. -/
def conflictBatch : Batch :=
  [{ device := "lighting", op := "set_brightness", args := [("level", 30)] },
    { device := "lighting", op := "set_brightness", args := [("level", 80)] }]

/-- The planner invented a device that has no card. -/
def unknownDeviceBatch : Batch :=
  [{ device := "computer", op := "open_file_explorer", args := [] }]

/-- Brightness level outside `[0, 100]`. -/
def outOfRangeBatch : Batch :=
  [{ device := "lighting", op := "set_brightness", args := [("level", 150)] }]

/-- Argument missing entirely. -/
def missingArgBatch : Batch :=
  [{ device := "lighting", op := "set_brightness", args := [] }]

#guard check cards state goodBatch
#guard !check cards state conflictBatch
#guard !check cards state unknownDeviceBatch
#guard !check cards state outOfRangeBatch
#guard !check cards state missingArgBatch
#guard !check cards stateLightsOffline goodBatch

end SmartHome.Examples
