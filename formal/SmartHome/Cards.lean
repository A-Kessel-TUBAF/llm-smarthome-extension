/-!
# Agent cards and tasks

The data model behind the allocation checker. An *agent card* declares, per device, which
operations it offers, what each operation needs (parameters, preconditions), touches
(reads/writes) and guarantees (postconditions). A *task* is what the planner emits: a device,
an operation and arguments. A *batch* is a list of tasks meant to run together.

Values are integers; booleans are encoded as `0`/`1` (e.g. `light.online == 1`).
-/

namespace SmartHome

/-- Name of a device agent, e.g. `"lighting"`. -/
abbrev DeviceId := String

/-- Name of an operation offered by a device, e.g. `"set_brightness"`. -/
abbrev OpName := String

/-- A state variable of the home, e.g. `"light.brightness"`. -/
abbrev StateVar := String

/-- An integer parameter of an operation together with its allowed range `[lo, hi]`. -/
structure Param where
  name : String
  lo : Int
  hi : Int
  deriving Repr, DecidableEq

/-- A condition on the home state, used for pre- and postconditions. -/
inductive Cond where
  /-- `var == c` for a constant `c`, e.g. `light.online == 1`. -/
  | eqConst (var : StateVar) (c : Int)
  /-- `var == p` where `p` is a parameter of the operation, e.g. `light.brightness == level`. -/
  | eqParam (var : StateVar) (p : String)
  deriving Repr, DecidableEq

/-- One operation of an agent card. -/
structure Operation where
  name : OpName
  params : List Param
  reads : List StateVar
  writes : List StateVar
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
  args : List (String × Int)
  deriving Repr

/-- A batch of tasks intended to run together (in parallel). -/
abbrev Batch := List Task

/-- The observed home state: the value of each known state variable. -/
abbrev State := List (StateVar × Int)

end SmartHome
