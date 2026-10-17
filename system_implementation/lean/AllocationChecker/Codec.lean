/-
  Runtime JSON boundary of the checker executable.

  NOT covered by `checkAllocation_sound`: JSON parsing (Lean's `Lean.Json`
  parser), decoding into the finite model, diagnostics, authorization bindings,
  response encoding and IO. The verdict field of every response is computed by
  `checkAllocation` exactly; everything else is integration code that is tested
  from Python (see `docs/trust-boundaries.md`).

  Exception: typed argument values. If the catalogue declares the types of an
  operation's arguments (`"types"`), the arguments of its commands and the literal
  arguments of its tasks are decoded by `TypedValues.decodeTypedArgs`, which is
  proven exact (`TypedValues.decodeTypedArgs_iff`): a wrongly typed value makes the
  request invalid instead of being encoded as some number.
-/
import Lean.Data.Json
import AllocationChecker.Check
import AllocationChecker.TypedValues

namespace AllocationChecker.Codec

open Lean (Json JsonNumber)
open AllocationChecker
open AllocationChecker.TypedValues (ArgType decodeTyped decodeTypedArgs)

def schemaRequest : String := "dexa-alloc-request/1"
def schemaResponse : String := "dexa-alloc-response/1"
def checkerRelease : String := "dexa-alloc-checker/0.1.0"
def specVersion : String := "allocation-spec/1"

/-! ### Bounds enforced while decoding -/

def maxId : Nat := 1000000
def maxTime : Nat := 1000000000000
def maxAbsValue : Nat := 1000000000
def maxSmallList : Nat := 64
def maxTasks : Nat := 4096
def maxCommands : Nat := 64
def maxOps : Nat := 1024
def maxDevices : Nat := 4096
def maxState : Nat := 65536
def maxEvidence : Nat := 8192
def maxActive : Nat := 1024

abbrev DecodeM := Except String

def field (j : Json) (k : String) : DecodeM Json :=
  match j.getObjVal? k with
  | .ok v => pure v
  | .error _ => throw s!"missing field '{k}'"

/-- Reject objects with fields outside the schema. -/
def onlyKeys (j : Json) (ctx : String) (allowed : List String) : DecodeM Unit := do
  match j.getObj? with
  | .ok kvs =>
    for k in kvs.keys do
      unless allowed.contains k do throw s!"{ctx}: unexpected field '{k}'"
  | .error _ => throw s!"{ctx}: object expected"

def asNat (j : Json) (ctx : String) (bound : Nat := maxTime) : DecodeM Nat :=
  match j.getNat? with
  | .ok n => if n ≤ bound then pure n else throw s!"{ctx}: value {n} exceeds bound {bound}"
  | .error _ => throw s!"{ctx}: natural number expected"

def asId (j : Json) (ctx : String) : DecodeM Nat := asNat j ctx maxId

def asInt (j : Json) (ctx : String) : DecodeM Int :=
  match j.getInt? with
  | .ok i => if i.natAbs ≤ maxAbsValue then pure i else throw s!"{ctx}: integer out of range"
  | .error _ => throw s!"{ctx}: integer expected"

def asBool (j : Json) (ctx : String) : DecodeM Bool :=
  match j.getBool? with
  | .ok b => pure b
  | .error _ => throw s!"{ctx}: boolean expected"

def asStr (j : Json) (ctx : String) : DecodeM String :=
  match j.getStr? with
  | .ok s => pure s
  | .error _ => throw s!"{ctx}: string expected"

def asArr (j : Json) (ctx : String) (bound : Nat := maxSmallList) : DecodeM (Array Json) :=
  match j.getArr? with
  | .ok a => if a.size ≤ bound then pure a else throw s!"{ctx}: list longer than {bound}"
  | .error _ => throw s!"{ctx}: array expected"

def listOf (j : Json) (ctx : String) (bound : Nat) (f : Json → String → DecodeM α) :
    DecodeM (List α) := do
  let a ← asArr j ctx bound
  let mut out : Array α := #[]
  for h : i in [0:a.size] do
    out := out.push (← f a[i] s!"{ctx}[{i}]")
  pure out.toList

def idList (j : Json) (ctx : String) : DecodeM (List Nat) := listOf j ctx maxSmallList asId

/-- A state key is encoded as `[dev, attr]`. -/
def decodeKey (j : Json) (ctx : String) : DecodeM StateKey := do
  let a ← asArr j ctx 2
  unless a.size = 2 do throw s!"{ctx}: key must be [dev, attr]"
  pure { dev := ← asId a[0]! ctx, attr := ← asId a[1]! ctx }

def decodeCmp (j : Json) (ctx : String) : DecodeM Cmp := do
  match ← asStr j ctx with
  | "eq" => pure .eq
  | "ne" => pure .ne
  | "le" => pure .le
  | "ge" => pure .ge
  | s => throw s!"{ctx}: unknown comparison '{s}'"

def decodeStatus (j : Json) (ctx : String) : DecodeM Status := do
  match ← asStr j ctx with
  | "pending" => pure .pending
  | "running" => pure .running
  | "verified" => pure .verified
  | "failed" => pure .failed
  | "invalidated" => pure .invalidated
  | s => throw s!"{ctx}: unknown status '{s}'"

/-- Precondition `[[dev, attr], cmp, value]`. -/
def decodePre (j : Json) (ctx : String) : DecodeM Precondition := do
  let a ← asArr j ctx 3
  unless a.size = 3 do throw s!"{ctx}: precondition must be [key, cmp, value]"
  pure { key := ← decodeKey a[0]! ctx, cmp := ← decodeCmp a[1]! ctx, val := ← asInt a[2]! ctx }

/-- Intrinsic operation precondition `[attr, cmp, value]`. -/
def decodeAttrPre (j : Json) (ctx : String) : DecodeM AttrPre := do
  let a ← asArr j ctx 3
  unless a.size = 3 do throw s!"{ctx}: precondition must be [attr, cmp, value]"
  pure { attr := ← asId a[0]! ctx, cmp := ← decodeCmp a[1]! ctx, val := ← asInt a[2]! ctx }

def decodeBound (j : Json) (ctx : String) : DecodeM ArgBound := do
  let a ← asArr j ctx 2
  unless a.size = 2 do throw s!"{ctx}: bound must be [lo, hi]"
  pure { lo := ← asInt a[0]! ctx, hi := ← asInt a[1]! ctx }

/-- `{"lit": v}` or `{"ref": task, "offset": o}`. -/
def decodeArgSrc (j : Json) (ctx : String) : DecodeM ArgSrc := do
  match j.getObjVal? "lit" with
  | .ok v =>
    onlyKeys j ctx ["lit"]
    pure (.lit (← asInt v ctx))
  | .error _ =>
    onlyKeys j ctx ["ref", "offset"]
    pure (.ref (← asId (← field j "ref") ctx) (← asInt (← field j "offset") ctx))

/-- Argument type `"bool"`, `{"number": scale}` or `{"enum": ["a", "b", ...]}`. -/
def decodeArgType (j : Json) (ctx : String) : DecodeM ArgType := do
  if j == .str "bool" then return .bool
  match j.getObjVal? "number", j.getObjVal? "enum" with
  | .ok sc, _ =>
    onlyKeys j ctx ["number"]
    let scale ← asNat sc ctx maxAbsValue
    unless 1 ≤ scale do throw s!"{ctx}: scale must be at least 1"
    pure (.number scale)
  | _, .ok vs =>
    onlyKeys j ctx ["enum"]
    let values ← listOf vs ctx maxSmallList asStr
    if values.isEmpty then throw s!"{ctx}: empty enumeration"
    pure (.enum values)
  | _, _ => throw (ctx ++ ": type must be \"bool\", {\"number\": scale} or {\"enum\": [...]}")

/-- An operation, and the declared types of its arguments if the catalogue gives them. -/
def decodeOp (j : Json) (ctx : String) : DecodeM (OpSpec × Option (List ArgType)) := do
  onlyKeys j ctx ["id", "cap", "bounds", "reads", "writes", "excl", "shared_reads", "pre",
    "coupled", "load_base", "load_per_unit", "types"]
  let op : OpSpec :=
    { id := ← asId (← field j "id") ctx
      cap := ← asId (← field j "cap") ctx
      bounds := ← listOf (← field j "bounds") s!"{ctx}.bounds" maxSmallList decodeBound
      reads := ← idList (← field j "reads") s!"{ctx}.reads"
      writes := ← idList (← field j "writes") s!"{ctx}.writes"
      excl := ← idList (← field j "excl") s!"{ctx}.excl"
      sharedReads := ← listOf (← field j "shared_reads") s!"{ctx}.shared_reads" maxSmallList decodeKey
      pre := ← listOf (← field j "pre") s!"{ctx}.pre" maxSmallList decodeAttrPre
      coupled := ← asBool (← field j "coupled") ctx
      loadBase := ← asInt (← field j "load_base") ctx
      loadPerUnit := ← asInt (← field j "load_per_unit") ctx }
  match j.getObjVal? "types" with
  | .ok ts =>
    let types ← listOf ts s!"{ctx}.types" maxSmallList decodeArgType
    unless types.length = op.bounds.length do
      throw s!"{ctx}: one type per argument bound expected"
    pure (op, some types)
  | .error _ => pure (op, none)

/-- Declared argument types per operation identifier. -/
abbrev TypeTable := List (Nat × List ArgType)

def typesOf (tt : TypeTable) (op : Nat) : Option (List ArgType) :=
  (tt.find? (·.1 == op)).map (·.2)

/-- Arguments of operation `op`: typed values if its types are declared, integers otherwise. -/
def decodeArgs (tt : TypeTable) (op : Nat) (j : Json) (ctx : String) : DecodeM (List Int) := do
  match typesOf tt op with
  | some types =>
    let a ← asArr j ctx
    match decodeTypedArgs types a.toList with
    | .ok ns => pure ns
    | .error e => throw s!"{ctx}: {e}"
  | none => listOf j ctx maxSmallList asInt

def decodeDevice (j : Json) (ctx : String) : DecodeM Device := do
  onlyKeys j ctx ["id", "caps", "ops"]
  pure
    { id := ← asId (← field j "id") ctx
      caps := ← idList (← field j "caps") s!"{ctx}.caps"
      ops := ← listOf (← field j "ops") s!"{ctx}.ops" maxOps asId }

def decodeCatalogue (j : Json) : DecodeM (Catalogue × TypeTable) := do
  onlyKeys j "catalogue" ["revision", "ops", "devices", "load_attr", "meter_key", "load_budget"]
  let ops ← listOf (← field j "ops") "catalogue.ops" maxOps decodeOp
  let cat : Catalogue :=
    { revision := ← asNat (← field j "revision") "catalogue.revision"
      ops := ops.map (·.1)
      devices := ← listOf (← field j "devices") "catalogue.devices" maxDevices decodeDevice
      loadAttr := ← asId (← field j "load_attr") "catalogue.load_attr"
      meterKey := ← decodeKey (← field j "meter_key") "catalogue.meter_key"
      loadBudget := ← asInt (← field j "load_budget") "catalogue.load_budget" }
  pure (cat, ops.filterMap fun (o, ts) => ts.map (o.id, ·))

/-- The task contract's arguments. With declared types, every literal is a typed value of its
    position's type; references to dependency outputs stay as they are. An empty list is
    accepted for any task: the arguments of tasks outside the batch do not matter, and a task
    of the batch without arguments fails the arity check of the specification. -/
def decodeTaskArgs (tt : TypeTable) (op : Nat) (j : Json) (ctx : String) :
    DecodeM (List ArgSrc) := do
  match typesOf tt op with
  | none => listOf j ctx maxSmallList decodeArgSrc
  | some types =>
    let a ← asArr j ctx
    if a.isEmpty then return []
    unless a.size = types.length do throw s!"{ctx}: one argument per declared type expected"
    (a.toList.zip types).mapM fun (arg, ty) => do
      match arg.getObjVal? "lit" with
      | .ok v =>
        onlyKeys arg ctx ["lit"]
        match decodeTyped ty v with
        | .ok n => pure (.lit n)
        | .error e => throw s!"{ctx}: {e}"
      | .error _ => decodeArgSrc arg ctx

def decodeTask (tt : TypeTable) (j : Json) (ctx : String) : DecodeM Task := do
  onlyKeys j ctx ["id", "status", "op", "device", "args", "deps", "rank", "pre", "deadline"]
  let op ← asId (← field j "op") ctx
  pure
    { id := ← asId (← field j "id") ctx
      status := ← decodeStatus (← field j "status") ctx
      op := op
      device := ← asId (← field j "device") ctx
      args := ← decodeTaskArgs tt op (← field j "args") s!"{ctx}.args"
      deps := ← idList (← field j "deps") s!"{ctx}.deps"
      rank := ← asNat (← field j "rank") ctx maxTasks
      pre := ← listOf (← field j "pre") s!"{ctx}.pre" maxSmallList decodePre
      deadline := ← asNat (← field j "deadline") ctx }

def decodeGraph (tt : TypeTable) (j : Json) : DecodeM TaskGraph := do
  onlyKeys j "graph" ["revision", "tasks"]
  pure
    { revision := ← asNat (← field j "revision") "graph.revision"
      tasks := ← listOf (← field j "tasks") "graph.tasks" maxTasks (decodeTask tt) }

/-- State entry `[dev, attr, value, version]`. -/
def decodeEntry (j : Json) (ctx : String) : DecodeM Entry := do
  let a ← asArr j ctx 4
  unless a.size = 4 do throw s!"{ctx}: entry must be [dev, attr, value, version]"
  pure { key := { dev := ← asId a[0]! ctx, attr := ← asId a[1]! ctx }
         val := ← asInt a[2]! ctx, ver := ← asNat a[3]! ctx }

/-- Basis entry `[dev, attr, version]`. -/
def decodeBasis (j : Json) (ctx : String) : DecodeM (StateKey × Nat) := do
  let a ← asArr j ctx 3
  unless a.size = 3 do throw s!"{ctx}: basis must be [dev, attr, version]"
  pure ({ dev := ← asId a[0]! ctx, attr := ← asId a[1]! ctx }, ← asNat a[2]! ctx)

def decodeEvidence (j : Json) (ctx : String) : DecodeM Evidence := do
  onlyKeys j ctx ["task", "verified", "valid_until", "output", "basis"]
  let out ← field j "output"
  pure
    { task := ← asId (← field j "task") ctx
      verified := ← asBool (← field j "verified") ctx
      validUntil := ← asNat (← field j "valid_until") ctx
      output := ← (if out.isNull then pure none else some <$> asInt out ctx)
      basis := ← listOf (← field j "basis") s!"{ctx}.basis" maxSmallList decodeBasis }

def decodeFootprint (j : Json) (ctx : String) : DecodeM Footprint := do
  onlyKeys j ctx ["reads", "writes", "excl"]
  pure
    { reads := ← listOf (← field j "reads") s!"{ctx}.reads" maxSmallList decodeKey
      writes := ← listOf (← field j "writes") s!"{ctx}.writes" maxSmallList decodeKey
      excl := ← listOf (← field j "excl") s!"{ctx}.excl" maxSmallList decodeKey }

def decodeSnapshot (j : Json) : DecodeM Snapshot := do
  onlyKeys j "snapshot" ["now", "worker_cap", "state", "evidence", "active"]
  pure
    { now := ← asNat (← field j "now") "snapshot.now"
      workerCap := ← asNat (← field j "worker_cap") "snapshot.worker_cap" maxCommands
      state := ← listOf (← field j "state") "snapshot.state" maxState decodeEntry
      evidence := ← listOf (← field j "evidence") "snapshot.evidence" maxEvidence decodeEvidence
      active := ← listOf (← field j "active") "snapshot.active" maxActive decodeFootprint }

def decodeCommand (tt : TypeTable) (j : Json) (ctx : String) : DecodeM Command := do
  onlyKeys j ctx ["task", "device", "op", "args"]
  let op ← asId (← field j "op") ctx
  pure
    { task := ← asId (← field j "task") ctx
      device := ← asId (← field j "device") ctx
      op := op
      args := ← decodeArgs tt op (← field j "args") s!"{ctx}.args" }

def decodeBatch (tt : TypeTable) (j : Json) : DecodeM AllocationBatch := do
  onlyKeys j "batch" ["graph_revision", "catalogue_revision", "commands"]
  pure
    { graphRev := ← asNat (← field j "graph_revision") "batch.graph_revision"
      catalogueRev := ← asNat (← field j "catalogue_revision") "batch.catalogue_revision"
      commands := ← listOf (← field j "commands") "batch.commands" maxCommands
        (decodeCommand tt) }

structure Request where
  requestId : String
  catalogue : Catalogue
  graph     : TaskGraph
  snapshot  : Snapshot
  batch     : AllocationBatch
  authTtl   : Nat

def decodeRequest (j : Json) : DecodeM Request := do
  onlyKeys j "request" ["schema", "request_id", "catalogue", "graph", "snapshot", "batch", "auth_ttl"]
  let schema ← asStr (← field j "schema") "schema"
  unless schema = schemaRequest do throw s!"unsupported schema '{schema}'"
  let rid ← asStr (← field j "request_id") "request_id"
  unless rid.length ≤ 128 do throw "request_id too long"
  let (cat, tt) ← decodeCatalogue (← field j "catalogue")
  pure
    { requestId := rid
      catalogue := cat
      graph := ← decodeGraph tt (← field j "graph")
      snapshot := ← decodeSnapshot (← field j "snapshot")
      batch := ← decodeBatch tt (← field j "batch")
      authTtl := ← asNat (← field j "auth_ttl") "auth_ttl" }

/-- Best-effort request identifier for error responses. -/
def requestIdOf (j : Json) : Json :=
  match j.getObjVal? "request_id" with
  | .ok (.str s) => if s.length ≤ 128 then .str s else .null
  | _ => .null

/-! ### Diagnostics (informative; the verdict is `checkAllocation`) -/

structure Diag where
  code   : String
  cmd    : Option Nat := none
  other  : Option Nat := none
  task   : Option Nat := none

def fitDiags (cat : Catalogue) (s : Snapshot) (g : TaskGraph) (c : Command) (t : Task)
    (o : OpSpec) (d : Device) : List String :=
  let checks : List (String × Bool) :=
    [ ("ineligible_status", eligibleB t.status),
      ("op_mismatch", t.op == c.op),
      ("device_mismatch", t.device == c.device),
      ("missing_capability", decide (o.cap ∈ d.caps)),
      ("unsupported_operation", decide (o.id ∈ d.ops)),
      ("arity_mismatch", c.args.length == o.bounds.length && t.args.length == o.bounds.length),
      ("argument_out_of_bounds", (o.bounds.zip c.args).all (fun p => inBoundsB p.1 p.2)),
      ("unresolved_argument", (t.args.zip c.args).all (fun p => argResolvedB s t p.1 p.2)),
      ("dependency_not_satisfied", t.deps.all (depSatisfiedB s g)),
      ("task_precondition_failed", t.pre.all (preHoldsB s)),
      ("operation_precondition_failed", o.pre.all (fun p => preHoldsB s (p.at c.device))),
      ("deadline_passed", decide (s.now < t.deadline)),
      ("load_policy_violated", policyB cat s o c) ]
  checks.filterMap (fun (code, ok) => if ok then none else some code)

def diagnose (s : Snapshot) (cat : Catalogue) (g : TaskGraph) (b : AllocationBatch) : List Diag :=
  Id.run do
    let mut out : Array Diag := #[]
    unless checkCatalogueWF cat do out := out.push { code := "catalogue_ids_ambiguous" }
    unless checkSnapshotWF s do out := out.push { code := "snapshot_keys_ambiguous" }
    unless nodupB (g.tasks.map (·.id)) do out := out.push { code := "graph_ids_ambiguous" }
    for t in g.tasks do
      for d in t.deps do
        unless rankOk g t d do
          out := out.push { code := "rank_witness_violated", task := some t.id, other := some d }
    unless b.graphRev == g.revision do out := out.push { code := "graph_revision_mismatch" }
    unless b.catalogueRev == cat.revision do out := out.push { code := "catalogue_revision_mismatch" }
    unless decide (b.commands.length ≤ s.workerCap) do out := out.push { code := "worker_capacity_exceeded" }
    unless nodupB (b.commands.map (·.task)) do out := out.push { code := "duplicate_task_allocation" }
    let cs := b.commands.toArray
    for h : i in [0:cs.size] do
      let c := cs[i]
      match lookupTask g c.task, lookupOp cat c.op, lookupDevice cat c.device with
      | some t, some o, some d =>
        for code in fitDiags cat s g c t o d do
          out := out.push { code := code, cmd := some i, task := some c.task }
      | t?, o?, d? =>
        if t?.isNone then out := out.push { code := "unknown_task", cmd := some i, task := some c.task }
        if o?.isNone then out := out.push { code := "unknown_operation", cmd := some i, task := some c.task }
        if d?.isNone then out := out.push { code := "unknown_device", cmd := some i, task := some c.task }
      unless activeIndepB cat s c do
        out := out.push { code := "conflicts_with_running_operation", cmd := some i, task := some c.task }
      unless s.active.all (fun f => disjointB f.writes (taskPreKeys g c)) do
        out := out.push { code := "task_precondition_conflicts_with_running_operation",
                          cmd := some i, task := some c.task }
      for h2 : j in [i+1:cs.size] do
        unless pairIndepB cat c cs[j] do
          out := out.push { code := "resource_conflict", cmd := some i, other := some j, task := some c.task }
        unless keepsTaskPreB cat g c cs[j] && keepsTaskPreB cat g cs[j] c do
          out := out.push { code := "task_precondition_conflict", cmd := some i, other := some j,
                            task := some c.task }
    pure out.toList

/-! ### Authorization bindings for accepted batches -/

def versionOf (s : Snapshot) (k : StateKey) : Nat :=
  match lookupEntry s k with
  | some e => e.ver
  | none => 0

def dedupKeys (ks : List StateKey) : List StateKey :=
  ks.foldl (fun acc k => if acc.contains k then acc else acc ++ [k]) []

/-- Every state key whose version the dispatcher must re-validate atomically:
    task and operation preconditions, evidence bases of dependencies, policy keys
    and the read/write footprint. -/
def relevantKeys (s : Snapshot) (cat : Catalogue) (g : TaskGraph) (c : Command) : List StateKey :=
  match lookupTask g c.task, lookupOp cat c.op with
  | some t, some o =>
    let fp := opFootprint cat o c.device
    let evKeys := t.deps.flatMap (fun d =>
      (s.evidence.filter (fun ev => evidenceValidB s ev d)).flatMap (fun ev => ev.basis.map (·.1)))
    let polKeys := if o.coupled then [cat.meterKey, { dev := c.device, attr := cat.loadAttr }] else []
    t.pre.map (·.key) ++ o.pre.map (fun p => (p.at c.device).key) ++ evKeys ++ polKeys ++
      fp.reads ++ fp.writes
  | _, _ => []

/-- Authorization expiry: the earliest of `now + ttl`, task deadlines and the
    validity of every evidence record the batch relies on. -/
def expiry (s : Snapshot) (g : TaskGraph) (b : AllocationBatch) (ttl : Nat) : Nat :=
  b.commands.foldl (fun acc c =>
    match lookupTask g c.task with
    | some t =>
      let evMin := t.deps.foldl (fun a d =>
        (s.evidence.filter (fun ev => evidenceValidB s ev d)).foldl (fun a2 ev => min a2 ev.validUntil) a) acc
      min evMin t.deadline
    | none => acc) (s.now + ttl)

/-! ### Encoding -/

def natJ (n : Nat) : Json := Json.num (JsonNumber.fromNat n)
def intJ (i : Int) : Json := Json.num (JsonNumber.fromInt i)
def optNatJ : Option Nat → Json
  | some n => natJ n
  | none => .null

def encodeCommand (c : Command) : Json :=
  Json.mkObj [("task", natJ c.task), ("device", natJ c.device), ("op", natJ c.op),
    ("args", Json.arr (c.args.map intJ).toArray)]

def encodeDiag (d : Diag) : Json :=
  Json.mkObj [("code", .str d.code), ("cmd", optNatJ d.cmd), ("other", optNatJ d.other),
    ("task", optNatJ d.task)]

def timingJ (parse check diag : Nat) : Json :=
  Json.mkObj [("parse_ns", natJ parse), ("check_ns", natJ check), ("diag_ns", natJ diag)]

def errorResponse (rid : Json) (msg : String) (parseNs : Nat) : Json :=
  Json.mkObj
    [ ("schema", .str schemaResponse), ("request_id", rid), ("checker_release", .str checkerRelease),
      ("spec_version", .str specVersion), ("verdict", .str "error"),
      ("error", .str msg), ("timing", timingJ parseNs 0 0) ]

/-- Process one request line. Returns the compact response line. -/
def handle (line : String) : IO String := do
  let t0 ← IO.monoNanosNow
  match Json.parse line with
  | .error e =>
    let t1 ← IO.monoNanosNow
    pure (errorResponse .null s!"malformed JSON: {e}" (t1 - t0)).compress
  | .ok j =>
    match decodeRequest j with
    | .error e =>
      let t1 ← IO.monoNanosNow
      pure (errorResponse (requestIdOf j) s!"invalid request: {e}" (t1 - t0)).compress
    | .ok req =>
      let t1 ← IO.monoNanosNow
      let ok := checkAllocation req.snapshot req.catalogue req.graph req.batch
      -- Storing the verdict in a reference forces its evaluation before `t2`.
      let okRef ← IO.mkRef ok
      let t2 ← IO.monoNanosNow
      let ok ← okRef.get
      let diags := if ok then [] else diagnose req.snapshot req.catalogue req.graph req.batch
      let bindings :=
        if ok then
          dedupKeys (req.batch.commands.flatMap (relevantKeys req.snapshot req.catalogue req.graph))
        else []
      let exp := if ok then expiry req.snapshot req.graph req.batch req.authTtl else 0
      let diagsJ := Json.arr (diags.map encodeDiag).toArray
      let bindJ := Json.arr (bindings.map (fun k =>
        Json.arr #[natJ k.dev, natJ k.attr, natJ (versionOf req.snapshot k)])).toArray
      let auxRef ← IO.mkRef (diagsJ, bindJ, exp)
      let t3 ← IO.monoNanosNow
      let (diagsJ, bindJ, exp) ← auxRef.get
      let resp := Json.mkObj
        [ ("schema", .str schemaResponse), ("request_id", .str req.requestId),
          ("checker_release", .str checkerRelease), ("spec_version", .str specVersion),
          ("verdict", .str (if ok then "accept" else "reject")),
          ("graph_revision", natJ req.graph.revision),
          ("catalogue_revision", natJ req.catalogue.revision),
          ("snapshot_now", natJ req.snapshot.now),
          ("commands", Json.arr (req.batch.commands.map encodeCommand).toArray),
          ("bindings", bindJ), ("expires_at", natJ exp), ("diagnostics", diagsJ),
          ("timing", timingJ (t1 - t0) (t2 - t1) (t3 - t2)) ]
      let body := resp.compress
      let t4 ← IO.monoNanosNow
      -- Splice the encoding time into the already-encoded object.
      pure ((body.dropEnd 1).toString ++ s!",\"encode_ns\":{t4 - t3}}")

end AllocationChecker.Codec
