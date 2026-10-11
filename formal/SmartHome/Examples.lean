import SmartHome.Allocation

/-!
# Examples

The lighting card from the extension plan, a thermostat card, and a few batches that the
checker should accept or reject. The `#guard` lines are checked at build time.
-/

namespace SmartHome.Examples

/-- A room identifier, e.g. `room "bedroom"`. -/
def room (name : String) : Value := .id "RoomID" name

/-- The lighting agent card from the extension plan. `set_brightness` follows the example
exactly; the example lists `set_color_temperature` as a capability without details, so its
parameters and range here are illustrative, not from the plan. -/
def lighting : AgentCard :=
  { device := "lighting"
    ops := [
      { name := "set_brightness"
        params := [{ name := "room", ty := .id "RoomID" }, { name := "level", ty := .int 0 100 }]
        reads := ["light.online", "light.brightness"]
        writes := ["light.brightness"]
        pre := [.eqConst "light.online" (.bool true)]
        post := [.eqParam "light.brightness" "level"] },
      { name := "set_color_temperature"
        params := [{ name := "room", ty := .id "RoomID" },
          { name := "kelvin", ty := .int 2000 6500 }]
        reads := ["light.online"]
        writes := ["light.color_temperature"]
        pre := [.eqConst "light.online" (.bool true)]
        post := [.eqParam "light.color_temperature" "kelvin"] }] }

/-- A thermostat agent card in the same style. -/
def thermostat : AgentCard :=
  { device := "thermostat"
    ops := [
      { name := "set_temperature"
        params := [{ name := "celsius", ty := .int 5 30 }]
        reads := ["thermostat.online"]
        writes := ["thermostat.target"]
        pre := [.eqConst "thermostat.online" (.bool true)]
        post := [.eqParam "thermostat.target" "celsius"] }] }

def cards : List AgentCard := [lighting, thermostat]

/-- Observed state: both devices online. -/
def state : State :=
  [("light.online", .bool true), ("light.brightness", .int 70),
    ("thermostat.online", .bool true), ("thermostat.target", .int 21)]

/-- Observed state: the lights are offline. -/
def stateLightsOffline : State :=
  [("light.online", .bool false), ("light.brightness", .int 70),
    ("thermostat.online", .bool true), ("thermostat.target", .int 21)]

/-- Dim the bedroom lights and warm the room: two devices, no shared writes. -/
def goodBatch : Batch :=
  [{ device := "lighting", op := "set_brightness",
      args := [("room", room "bedroom"), ("level", .int 30)] },
    { device := "thermostat", op := "set_temperature", args := [("celsius", .int 23)] }]

/-- Two tasks write `light.brightness`. -/
def conflictBatch : Batch :=
  [{ device := "lighting", op := "set_brightness",
      args := [("room", room "bedroom"), ("level", .int 30)] },
    { device := "lighting", op := "set_brightness",
      args := [("room", room "bedroom"), ("level", .int 80)] }]

/-- The planner invented a device that has no card. -/
def unknownDeviceBatch : Batch :=
  [{ device := "computer", op := "open_file_explorer", args := [] }]

/-- Brightness level outside `[0, 100]`. -/
def outOfRangeBatch : Batch :=
  [{ device := "lighting", op := "set_brightness",
      args := [("room", room "bedroom"), ("level", .int 150)] }]

/-- The `room` argument is missing. -/
def missingArgBatch : Batch :=
  [{ device := "lighting", op := "set_brightness", args := [("level", .int 30)] }]

/-- The brightness level is given as a boolean instead of an integer. -/
def wrongTypeBatch : Batch :=
  [{ device := "lighting", op := "set_brightness",
      args := [("room", room "bedroom"), ("level", .bool true)] }]

/-- The `room` argument is an identifier of the wrong kind. -/
def wrongKindBatch : Batch :=
  [{ device := "lighting", op := "set_brightness",
      args := [("room", .id "DeviceID" "bedroom"), ("level", .int 30)] }]

#guard check cards state goodBatch
#guard !check cards state conflictBatch
#guard !check cards state unknownDeviceBatch
#guard !check cards state outOfRangeBatch
#guard !check cards state missingArgBatch
#guard !check cards state wrongTypeBatch
#guard !check cards state wrongKindBatch
#guard !check cards stateLightsOffline goodBatch

end SmartHome.Examples
