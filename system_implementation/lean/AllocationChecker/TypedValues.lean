/-
  Typed argument values at the JSON boundary.

  The checker's model represents every argument as an integer (`Types.lean`). The agents,
  however, produce typed values: numbers such as `21.5`, booleans, and enumeration values such
  as `"heat"`. When the catalogue declares the type of each argument of an operation
  (`ArgType`), the executable decodes the typed JSON value itself, instead of relying on an
  encoding done in Python:

  * a number is multiplied by the declared scale (`21.5` with scale 10 becomes `215`) and must
    then be a whole number within the checker's value range,
  * `true` / `false` become `1` / `0`,
  * an enumeration value becomes its position in the declared list.

  `decodeTyped_iff` proves that decoding succeeds exactly for the values of the declared type,
  and yields their encoding (`Encodes`); a wrongly typed value is always an error
  (`decodeTyped_wrong_type`). `decodeTypedArgs_iff` lifts this to argument lists. The
  specification, the checker and their proofs are unchanged: they see the same integers as
  before.

  This module imports only `Lean.Data.Json` (for the JSON values).
-/
import Lean.Data.Json

namespace AllocationChecker.TypedValues

open Lean (Json JsonNumber)

/-- Largest absolute value of an encoded argument (the same bound as `Codec.maxAbsValue`). -/
def maxAbs : Nat := 1000000000

/-- The declared type of an operation argument. -/
inductive ArgType where
  /-- A number, represented as the whole number `value * scale`, e.g. `21.5` °C with scale
      `10` as `215`. -/
  | number (scale : Nat)
  /-- A boolean, represented as `1` (`true`) or `0` (`false`). -/
  | bool
  /-- One of the listed strings, represented by its position in the list. -/
  | enum (values : List String)
  deriving Repr, DecidableEq, Inhabited

/-- `j` is a value of type `ty` that the checker represents by the integer `n`. -/
def Encodes : ArgType → Json → Int → Prop
  | .number scale, j, n =>
    ∃ x : JsonNumber, j = .num x ∧ n * 10 ^ x.exponent = x.mantissa * scale ∧
      n.natAbs ≤ maxAbs
  | .bool, j, n => ∃ b : Bool, j = .bool b ∧ n = if b then 1 else 0
  | .enum vs, j, n => ∃ s i, j = .str s ∧ vs.idxOf? s = some i ∧ n = i

/-- Decode a typed JSON value into the checker's integer representation. -/
def decodeTyped : ArgType → Json → Except String Int
  | .number scale, .num x =>
    if (x.mantissa * scale) % (10 ^ x.exponent : Int) = 0 then
      let n := x.mantissa * scale / 10 ^ x.exponent
      if n.natAbs ≤ maxAbs then .ok n else .error "number out of range"
    else .error s!"{x} is not a whole multiple of 1/{scale}"
  | .number _, _ => .error "number expected"
  | .bool, .bool b => .ok (if b then 1 else 0)
  | .bool, _ => .error "boolean expected"
  | .enum vs, .str s =>
    match vs.idxOf? s with
    | some i => .ok i
    | none => .error s!"'{s}' is not one of {vs}"
  | .enum _, _ => .error "string expected"

theorem pow10_ne_zero (e : Nat) : (10 : Int) ^ e ≠ 0 := by
  induction e with
  | zero => decide
  | succ e ih => rw [Int.pow_succ]; exact Int.mul_ne_zero ih (by decide)

/-- **Typed decoding is exact**: it succeeds exactly for the values of the declared type, and
yields their encoding. -/
theorem decodeTyped_iff {ty : ArgType} {j : Json} {n : Int} :
    decodeTyped ty j = .ok n ↔ Encodes ty j n := by
  cases ty with
  | number scale =>
    cases j with
    | num x =>
      have hp := pow10_ne_zero x.exponent
      simp only [decodeTyped, Encodes, Json.num.injEq, exists_eq_left']
      constructor
      · intro h
        split at h
        · rename_i hm
          split at h
          · rename_i hb
            cases h
            exact ⟨Int.ediv_mul_cancel (Int.dvd_of_emod_eq_zero hm), hb⟩
          · cases h
        · cases h
      · rintro ⟨hn, hb⟩
        have hm : x.mantissa * scale % 10 ^ x.exponent = 0 := by
          rw [← hn]; exact Int.mul_emod_left _ _
        have hd : x.mantissa * scale / 10 ^ x.exponent = n := by
          rw [← hn]; exact Int.mul_ediv_cancel _ hp
        simp [hm, hd, hb]
    | _ => simp [decodeTyped, Encodes]
  | bool =>
    cases j with
    | bool b => simp [decodeTyped, Encodes, eq_comm]
    | _ => simp [decodeTyped, Encodes]
  | enum vs =>
    cases j with
    | str s =>
      simp only [decodeTyped, Encodes, Json.str.injEq]
      cases h : vs.idxOf? s <;> simp [h, eq_comm]
    | _ => simp [decodeTyped, Encodes]

/-- **Wrongly typed values are rejected**: a value that does not have the declared type is
always a decoding error, never a number. -/
theorem decodeTyped_wrong_type {ty : ArgType} {j : Json} (h : ¬ ∃ n, Encodes ty j n) :
    ∃ e, decodeTyped ty j = .error e := by
  cases hd : decodeTyped ty j with
  | error e => exact ⟨e, rfl⟩
  | ok n => exact absurd ⟨n, decodeTyped_iff.mp hd⟩ h

/-- Decode an argument list against the declared types, position by position. -/
def decodeTypedArgs : List ArgType → List Json → Except String (List Int)
  | [], [] => .ok []
  | ty :: tys, j :: js =>
    match decodeTyped ty j, decodeTypedArgs tys js with
    | .ok n, .ok ns => .ok (n :: ns)
    | .error e, _ => .error e
    | _, .error e => .error e
  | _, _ => .error "wrong number of arguments"

/-- Every argument has its declared type and the given encoding. -/
def AllEncode : List ArgType → List Json → List Int → Prop
  | [], [], [] => True
  | ty :: tys, j :: js, n :: ns => Encodes ty j n ∧ AllEncode tys js ns
  | _, _, _ => False

/-- Decoding an argument list succeeds exactly when every argument has its declared type. -/
theorem decodeTypedArgs_iff : ∀ {tys : List ArgType} {js : List Json} {ns : List Int},
    decodeTypedArgs tys js = .ok ns ↔ AllEncode tys js ns
  | [], [], ns => by cases ns <;> simp [decodeTypedArgs, AllEncode]
  | [], _ :: _, ns => by simp [decodeTypedArgs, AllEncode]
  | _ :: _, [], ns => by simp [decodeTypedArgs, AllEncode]
  | ty :: tys, j :: js, ns => by
    have ih := @decodeTypedArgs_iff tys js
    cases h₁ : decodeTyped ty j with
    | error e =>
      cases ns with
      | nil => simp [decodeTypedArgs, AllEncode, h₁]
      | cons n ns =>
        simp only [decodeTypedArgs, AllEncode, h₁]
        constructor
        · intro h; cases h
        · rintro ⟨hn, -⟩; rw [decodeTyped_iff.mpr hn] at h₁; cases h₁
    | ok n =>
      cases h₂ : decodeTypedArgs tys js with
      | error e =>
        cases ns with
        | nil => simp [decodeTypedArgs, AllEncode, h₁, h₂]
        | cons m ms =>
          simp only [decodeTypedArgs, AllEncode, h₁, h₂]
          constructor
          · intro h; cases h
          · rintro ⟨-, hs⟩; rw [ih.mpr hs] at h₂; cases h₂
      | ok ms =>
        have hn := decodeTyped_iff.mp h₁
        have hs := ih.mp h₂
        cases ns with
        | nil => simp [decodeTypedArgs, AllEncode, h₁, h₂]
        | cons n' ns' =>
          simp only [decodeTypedArgs, AllEncode, h₁, h₂, Except.ok.injEq, List.cons.injEq]
          constructor
          · rintro ⟨rfl, rfl⟩; exact ⟨hn, hs⟩
          · rintro ⟨hn', hs'⟩
            have e1 := decodeTyped_iff.mpr hn'
            have e2 := ih.mpr hs'
            rw [h₁] at e1; rw [h₂] at e2
            cases e1; cases e2
            exact ⟨rfl, rfl⟩

/-! ### Examples (checked by the kernel) -/

/-- The thermostat's `set_temperature(temperature_c)` with 0.1 °C steps, and `set_mode(mode)`. -/
def celsius : ArgType := .number 10
def mode : ArgType := .enum ["off", "heat", "cool", "auto"]

example : (decodeTyped celsius (.num ⟨215, 1⟩)).toOption = some 215 := by decide  -- 21.5
example : (decodeTyped celsius (.num ⟨22, 0⟩)).toOption = some 220 := by decide   -- 22
example : (decodeTyped mode (.str "heat")).toOption = some 1 := by decide
example : (decodeTyped .bool (.bool true)).toOption = some 1 := by decide
-- 21.55 is not a multiple of 0.1
example : (decodeTyped celsius (.num ⟨2155, 2⟩)).isOk = false := by decide
example : (decodeTyped celsius (.str "warm")).isOk = false := by decide  -- text, not a number
example : (decodeTyped mode (.str "warm")).isOk = false := by decide     -- not a mode
example : (decodeTyped mode (.num ⟨1, 0⟩)).isOk = false := by decide     -- a number, not a mode
example : (decodeTyped .bool (.num ⟨1, 0⟩)).isOk = false := by decide    -- 1, not a boolean
example : (decodeTypedArgs [mode, celsius] [.str "cool", .num ⟨19, 0⟩]).toOption =
    some [2, 190] := by decide
example : (decodeTypedArgs [mode, celsius] [.str "cool"]).isOk = false := by decide

end AllocationChecker.TypedValues
