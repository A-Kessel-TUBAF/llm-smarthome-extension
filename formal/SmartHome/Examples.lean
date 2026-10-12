import SmartHome.Allocation

/-!
# Examples

A thermostat card, the lighting card from the extension plan, and a few batches that the
checker should accept or reject. The `#guard` lines are checked at build time.
-/

namespace SmartHome.Examples

/-! ## State variables -/

/-- Property `p` of device `d` in room `r`. -/
def var (d : DeviceId) (r : RoomId) (p : String) : StateVar :=
  { device := d, room := some r, prop := p }

/-- The temperature setting of the thermostat in the bedroom. -/
def bedroomTemperature : StateVar := var "thermostat" "bedroom" "temperature"

/-- Whether the thermostat in the bedroom is online. -/
def bedroomThermostatOnline : StateVar := var "thermostat" "bedroom" "online"

/-! ## Agent cards -/

/-- A room identifier, e.g. `room "bedroom"`. -/
def room (name : String) : Value := .id "RoomID" name

/-- On a card: property `p` in the room given by the task's `room` argument. -/
def inRoom (p : String) : VarRef := { prop := p, room := .param "room" }

/-- A thermostat agent card: set the temperature in a room. -/
def thermostat : AgentCard :=
  { device := "thermostat"
    ops := [
      { name := "set_temperature"
        params := [{ name := "room", ty := .id "RoomID" }, { name := "celsius", ty := .int 5 30 }]
        reads := [inRoom "online", inRoom "temperature"]
        writes := [inRoom "temperature"]
        pre := [.eqConst (inRoom "online") (.bool true)]
        post := [.eqParam (inRoom "temperature") "celsius"] }] }

/-- The lighting agent card from the extension plan. `set_brightness` follows the example
exactly; the example lists `set_color_temperature` as a capability without details, so its
parameters and range here are illustrative, not from the plan. -/
def lighting : AgentCard :=
  { device := "lighting"
    ops := [
      { name := "set_brightness"
        params := [{ name := "room", ty := .id "RoomID" }, { name := "level", ty := .int 0 100 }]
        reads := [inRoom "online", inRoom "brightness"]
        writes := [inRoom "brightness"]
        pre := [.eqConst (inRoom "online") (.bool true)]
        post := [.eqParam (inRoom "brightness") "level"] },
      { name := "set_color_temperature"
        params := [{ name := "room", ty := .id "RoomID" },
          { name := "kelvin", ty := .int 2000 6500 }]
        reads := [inRoom "online"]
        writes := [inRoom "color_temperature"]
        pre := [.eqConst (inRoom "online") (.bool true)]
        post := [.eqParam (inRoom "color_temperature") "kelvin"] }] }

def cards : List AgentCard := [thermostat, lighting]

/-! ## Observed states -/

/-- Home state when there is a thermostat in the bedroom,
some lights in the bedroom and living room, all online. -/
def state : HomeState :=
  [(bedroomThermostatOnline, .bool true), (bedroomTemperature, .int 21),
    (var "lighting" "bedroom" "online", .bool true),
    (var "lighting" "bedroom" "brightness", .int 70),
    (var "lighting" "living_room" "online", .bool true),
    (var "lighting" "living_room" "brightness", .int 40)]

/-- The same home, but the bedroom thermostat is offline. -/
def stateThermostatOffline : HomeState :=
  (bedroomThermostatOnline, .bool false) :: state.tail

/-! ## Tasks and batches -/

/-- Set the bedroom thermostat to `c` °C (warming or cooling). -/
def setBedroomTemperature (c : Int) : Task :=
  { device := "thermostat", op := "set_temperature",
    args := [("room", room "bedroom"), ("celsius", .int c)] }

/-- Set brightness of the lights in room `r` to `l` % (dimming or brightening). -/
def setBrightness (r : String) (l : Int) : Task :=
  { device := "lighting", op := "set_brightness", args := [("room", room r), ("level", .int l)] }

/-- The thermostat task writes exactly `bedroomTemperature`. -/
example : writesOf cards (setBedroomTemperature 23) = [bedroomTemperature] := rfl

/-- Set the bedroom temperature and dim its lights: two devices, no shared writes. -/
def goodBatch : Batch := [setBedroomTemperature 23, setBrightness "bedroom" 30]

/-- Dim the bedroom and the living room: same device and property, different rooms. -/
def twoRoomsBatch : Batch := [setBrightness "bedroom" 30, setBrightness "living_room" 0]

/-- Two tasks set the bedroom temperature. -/
def conflictBatch : Batch := [setBedroomTemperature 23, setBedroomTemperature 19]

/-- 35 °C is outside the thermostat's range `[5, 30]`. -/
def tooHotBatch : Batch := [setBedroomTemperature 35]

/-- The planner invented a device that has no card. -/
def unknownDeviceBatch : Batch := [{ device := "computer", op := "open_file_explorer", args := [] }]

/-- The `room` argument is missing. -/
def missingArgBatch : Batch :=
  [{ device := "thermostat", op := "set_temperature", args := [("celsius", .int 23)] }]

/-- The temperature is given as a boolean instead of an integer. -/
def wrongTypeBatch : Batch :=
  [{ device := "thermostat", op := "set_temperature",
      args := [("room", room "bedroom"), ("celsius", .bool true)] }]

/-- The `room` argument is an identifier of the wrong kind. -/
def wrongKindBatch : Batch :=
  [{ device := "thermostat", op := "set_temperature",
      args := [("room", .id "DeviceID" "bedroom"), ("celsius", .int 23)] }]

/-- There is no thermostat in the living room, so its precondition (`online == true`) fails. -/
def noSuchRoomBatch : Batch :=
  [{ device := "thermostat", op := "set_temperature",
      args := [("room", room "living_room"), ("celsius", .int 23)] }]

-- Accepted batches
#guard check cards state goodBatch
#guard check cards state twoRoomsBatch

-- Rejected batches
#guard !check cards state conflictBatch
#guard !check cards state tooHotBatch
#guard !check cards state unknownDeviceBatch
#guard !check cards state missingArgBatch
#guard !check cards state wrongTypeBatch
#guard !check cards state wrongKindBatch
#guard !check cards state noSuchRoomBatch
#guard !check cards stateThermostatOffline goodBatch

end SmartHome.Examples
