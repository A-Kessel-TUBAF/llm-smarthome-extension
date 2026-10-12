/-
  Kernel-checked examples. Every `decide` below is evaluated by the Lean kernel
  (no `native_decide`).

  Scenario (mirrors the offline demo): a search task `t0` has already observed
  the outdoor temperature (verified evidence, output 8). The bedroom thermostat
  task `t1` sets the setpoint to outdoor + 4 and depends on `t0`. Tasks `t2`
  (dim bedroom light) and `t3` (set living-room volume) are independent.
-/
import AllocationChecker.Completeness

namespace AllocationChecker.Examples

open AllocationChecker

/-! Attribute, device, capability and operation identifiers. -/
def aOnline : Nat := 0
def aBrightness : Nat := 1
def aSetpoint : Nat := 2
def aVolume : Nat := 3
def aLoad : Nat := 4
def aActuator : Nat := 5
def aResult : Nat := 6
def aTotal : Nat := 7

def dLight : Nat := 0
def dThermo : Nat := 1
def dAudio : Nat := 2
def dSearch : Nat := 3
def dMeter : Nat := 9

def cat : Catalogue :=
  { revision := 1
    loadAttr := aLoad
    meterKey := ⟨dMeter, aTotal⟩
    loadBudget := 3000
    ops :=
      [ { id := 0, cap := 0, bounds := [⟨0, 100⟩], reads := [], writes := [aBrightness]
          excl := [aActuator], sharedReads := [], pre := [⟨aOnline, .ge, 1⟩]
          coupled := false, loadBase := 0, loadPerUnit := 0 },
        { id := 1, cap := 1, bounds := [⟨10, 30⟩], reads := [], writes := [aSetpoint]
          excl := [aActuator], sharedReads := [], pre := [⟨aOnline, .ge, 1⟩]
          coupled := true, loadBase := 0, loadPerUnit := 100 },
        { id := 2, cap := 2, bounds := [⟨0, 100⟩], reads := [], writes := [aVolume]
          excl := [aActuator], sharedReads := [], pre := [⟨aOnline, .ge, 1⟩]
          coupled := false, loadBase := 0, loadPerUnit := 0 },
        { id := 3, cap := 3, bounds := [⟨0, 50⟩], reads := [], writes := [aResult]
          excl := [], sharedReads := [], pre := []
          coupled := false, loadBase := 0, loadPerUnit := 0 } ]
    devices :=
      [ ⟨dLight, [0], [0]⟩, ⟨dThermo, [1], [1]⟩, ⟨dAudio, [2], [2]⟩, ⟨dSearch, [3], [3]⟩ ] }

def mkTask (id : Nat) (st : Status) (op dev : Nat) (args : List ArgSrc) (deps : List Nat)
    (rank : Nat) : Task :=
  { id := id, status := st, op := op, device := dev, args := args, deps := deps, rank := rank
    pre := [], deadline := 1000 }

def graph : TaskGraph :=
  { revision := 7
    tasks :=
      [ mkTask 0 .verified 3 dSearch [.lit 7] [] 0,
        mkTask 1 .pending 1 dThermo [.ref 0 4] [0] 1,
        mkTask 2 .pending 0 dLight [.lit 30] [] 0,
        mkTask 3 .pending 2 dAudio [.lit 20] [] 0,
        mkTask 4 .pending 0 dLight [.lit 80] [] 0,
        mkTask 5 .running 2 dAudio [.lit 10] [] 0,
        mkTask 6 .pending 0 dLight [.lit 150] [] 0 ] }

def baseState (meter : Int) (resultVer : Nat) : List Entry :=
  [ ⟨⟨dLight, aOnline⟩, 1, 1⟩, ⟨⟨dLight, aBrightness⟩, 70, 3⟩,
    ⟨⟨dThermo, aOnline⟩, 1, 1⟩, ⟨⟨dThermo, aSetpoint⟩, 19, 2⟩, ⟨⟨dThermo, aLoad⟩, 800, 2⟩,
    ⟨⟨dAudio, aOnline⟩, 1, 1⟩, ⟨⟨dAudio, aVolume⟩, 50, 1⟩,
    ⟨⟨dSearch, aResult⟩, 8, resultVer⟩,
    ⟨⟨dMeter, aTotal⟩, meter, 4⟩ ]

def evidence0 : Evidence :=
  { task := 0, verified := true, validUntil := 500, output := some 8
    basis := [(⟨dSearch, aResult⟩, 5)] }

def mkSnap (now cap : Nat) (meter : Int) (resultVer : Nat) (active : List Footprint) : Snapshot :=
  { now := now, workerCap := cap, state := baseState meter resultVer, evidence := [evidence0]
    active := active }

/-- Nominal snapshot. -/
def snap : Snapshot := mkSnap 100 4 1500 5 []

def cmd (task dev op : Nat) (args : List Int) : Command :=
  { task := task, device := dev, op := op, args := args }

def batch (cs : List Command) : AllocationBatch :=
  { graphRev := 7, catalogueRev := 1, commands := cs }

def cDim : Command := cmd 2 dLight 0 [30]
def cVol : Command := cmd 3 dAudio 2 [20]
def cHeat : Command := cmd 1 dThermo 1 [12]

/-! ### Accepted batches -/

/-- A nonempty batch with two independent device actions. -/
def twoIndependent : AllocationBatch := batch [cDim, cVol]

theorem twoIndependent_nonempty : twoIndependent.commands.length = 2 := rfl

theorem twoIndependent_accepted : checkAllocation snap cat graph twoIndependent = true := by
  decide

theorem twoIndependent_valid : AllocationValid snap cat graph twoIndependent :=
  checkAllocation_sound _ _ _ _ twoIndependent_accepted

/-- Adds the dependent thermostat action whose argument is resolved from the
    verified search output (8 + 4 = 12) and which passes the load policy
    (1500 - 800 + 12 * 100 = 1900 ≤ 3000). -/
def withDependent : AllocationBatch := batch [cDim, cVol, cHeat]

theorem withDependent_accepted : checkAllocation snap cat graph withDependent = true := by
  decide

theorem withDependent_valid : AllocationValid snap cat graph withDependent :=
  checkAllocation_sound _ _ _ _ withDependent_accepted

/-! ### Rejected batches. Each rejection is a genuine specification violation
    by `not_valid_of_rejected` (completeness). -/

/-- Write/write and exclusive-actuator conflict on the bedroom light. -/
def conflicting : AllocationBatch := batch [cDim, cmd 4 dLight 0 [80]]
theorem conflicting_rejected : checkAllocation snap cat graph conflicting = false := by decide
theorem conflicting_invalid : ¬ AllocationValid snap cat graph conflicting :=
  not_valid_of_rejected conflicting_rejected

/-- The search result changed after the evidence was observed (version 5 → 6). -/
def staleSnap : Snapshot := mkSnap 100 4 1500 6 []
theorem stale_rejected : checkAllocation staleSnap cat graph withDependent = false := by decide
theorem stale_invalid : ¬ AllocationValid staleSnap cat graph withDependent :=
  not_valid_of_rejected stale_rejected

/-- The evidence expired (valid until 500, now 600). -/
def expiredSnap : Snapshot := mkSnap 600 4 1500 5 []
theorem expired_rejected : checkAllocation expiredSnap cat graph (batch [cHeat]) = false := by
  decide
theorem expired_invalid : ¬ AllocationValid expiredSnap cat graph (batch [cHeat]) :=
  not_valid_of_rejected expired_rejected

/-- Argument not resolved from the verified dependency output (15 ≠ 8 + 4). -/
theorem wrongInput_rejected :
    checkAllocation snap cat graph (batch [cmd 1 dThermo 1 [15]]) = false := by decide

/-- More commands than worker capacity. -/
def oneWorker : Snapshot := mkSnap 100 1 1500 5 []
theorem overCapacity_rejected : checkAllocation oneWorker cat graph twoIndependent = false := by
  decide

/-- A running task cannot be scheduled again. -/
theorem running_rejected : checkAllocation snap cat graph (batch [cmd 5 dAudio 2 [10]]) = false := by
  decide

/-- Household load policy: 2900 - 800 + 1200 = 3300 > 3000. -/
def highLoad : Snapshot := mkSnap 100 4 2900 5 []
theorem overBudget_rejected : checkAllocation highLoad cat graph (batch [cHeat]) = false := by
  decide

/-- A running operation still reserves the light's brightness key. -/
def busyLight : Snapshot :=
  mkSnap 100 4 1500 5 [{ reads := [], writes := [⟨dLight, aBrightness⟩], excl := [] }]
theorem activeConflict_rejected : checkAllocation busyLight cat graph (batch [cDim]) = false := by
  decide

/-- Argument outside the catalogue bounds (brightness 150). -/
theorem outOfBounds_rejected :
    checkAllocation snap cat graph (batch [cmd 6 dLight 0 [150]]) = false := by decide

/-- The same task allocated twice. -/
theorem duplicate_rejected : checkAllocation snap cat graph (batch [cDim, cDim]) = false := by
  decide

/-- Batch proposed for a different graph revision. -/
theorem staleRevision_rejected :
    checkAllocation snap cat graph { twoIndependent with graphRev := 6 } = false := by decide

/-- Wrong target device (the command addresses the thermostat for a light task). -/
theorem wrongDevice_rejected :
    checkAllocation snap cat graph (batch [cmd 2 dThermo 0 [30]]) = false := by decide

end AllocationChecker.Examples
