/-!
# Agent cards and tasks

The data model behind the allocation checker. An *agent card* declares, per device, which
operations it offers, what each operation needs (parameters, preconditions), touches
(reads/writes) and guarantees (postconditions). A *task* is what the planner emits: a device,
an operation and arguments. A *batch* is a list of tasks meant to run together.

Values are typed: booleans, integers, and identifiers of a given kind (e.g. a `RoomID`), so the
agent card example can be written as is, e.g. `light.online == true` and `room : RoomID`.

A *state variable* is one property of one device, optionally in one room, e.g. the temperature
setting of the thermostat in the bedroom. Cards refer to the variables of their own device only,
and may leave the room to a task argument (e.g. "the brightness in the room given by `room`").
-/

namespace SmartHome

/-- Name of a device agent, e.g. `"lighting"`. -/
abbrev DeviceId := String

/-- Name of an operation offered by a device, e.g. `"set_brightness"`. -/
abbrev OpName := String

/-- Name of a room, e.g. `"bedroom"`. -/
abbrev RoomId := String

/-- A state variable of the home: one property of one device, optionally in one room, e.g. the
temperature setting of the thermostat in the bedroom. -/
structure StateVar where
  device : DeviceId
  /-- The room, or `none` for a property of the whole home. -/
  room : Option RoomId
  prop : String
  deriving Repr, DecidableEq

/-- How an agent card refers to the room of a state variable. -/
inductive RoomRef where
  /-- The variable belongs to the whole home, not to a room. -/
  | home
  /-- A fixed room, e.g. `"bedroom"`. -/
  | fixed (r : RoomId)
  /-- The room given by the task's argument named `p`, e.g. `room`. -/
  | param (p : String)
  deriving Repr, DecidableEq

/-- How an agent card refers to a state variable of its own device: the property, and where the
room comes from. For example, `{ prop := "brightness", room := .param "room" }` is the brightness
in the room the task names. -/
structure VarRef where
  prop : String
  room : RoomRef
  deriving Repr, DecidableEq

/-- A value of a state variable or of a task argument. -/
inductive Value where
  /-- A boolean, e.g. the value of `light.online`. -/
  | bool (b : Bool)
  /-- An integer, e.g. a brightness level or a temperature. -/
  | int (n : Int)
  /-- An identifier `name` of kind `kind`, e.g. `id "RoomID" "bedroom"`. -/
  | id (kind : String) (name : String)
  deriving Repr, DecidableEq

/-- The type of an operation parameter. -/
inductive ParamType where
  /-- A boolean. -/
  | bool
  /-- An integer in the range `[lo, hi]`, written `Integer[lo..hi]` on agent cards. -/
  | int (lo hi : Int)
  /-- An identifier of the given kind, e.g. `RoomID`. -/
  | id (kind : String)
  deriving Repr, DecidableEq

/-- A parameter of an operation together with its type. -/
structure Param where
  name : String
  ty : ParamType
  deriving Repr, DecidableEq

/-- A condition on the home state, used for pre- and postconditions. -/
inductive Cond where
  /-- `var == c` for a constant `c`, e.g. `light.online == true`. -/
  | eqConst (var : VarRef) (c : Value)
  /-- `var == p` where `p` is a parameter of the operation, e.g. `light.brightness == level`. -/
  | eqParam (var : VarRef) (p : String)
  deriving Repr, DecidableEq

/-- One operation of an agent card. -/
structure Operation where
  name : OpName
  params : List Param
  reads : List VarRef
  writes : List VarRef
  pre : List Cond
  post : List Cond
  deriving Repr

/-- An agent card: a device and the operations it offers. -/
structure AgentCard where
  device : DeviceId
  ops : List Operation
  deriving Repr

/-- A task as produced by the planner. -/
structure Task where
  device : DeviceId
  op : OpName
  args : List (String × Value)
  deriving Repr

/-- A batch of tasks intended to run together (in parallel). -/
abbrev Batch := List Task

/-- The observed home state: the value of each known state variable. -/
abbrev State := List (StateVar × Value)

end SmartHome
