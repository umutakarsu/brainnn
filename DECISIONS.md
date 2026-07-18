# DECISIONS

Decisions made during implementation that the HANDOFF spec did not fully cover.
Per §9, the rule is: make the reasonable call, write it here, keep moving.

---

## D1 — Where the product lives

The session harness handed a boilerplate instruction to develop on a branch
inside `brainnn`. That directly conflicts with the HANDOFF (§2, §5) and Umut's
explicit instruction to build in a **new** repo `umutakarsu/loop`. Umut's direct
instruction wins: the product is built in `umutakarsu/loop`, standalone, with no
import from `brainnn`. `brainnn` is left untouched.

## D2 — Python package name

`loop` is a reserved-ish word and shadows nothing in stdlib, so the package is
named `loop/` as the spec's tree shows. The Streamlit entry point stays at repo
root as `app.py`.

## D3 — Branding constant

Per §2 ("leave branding in one config constant so it is a one-line change"), the
product name lives as `APP_NAME` in `loop/copy.py`. Changing the name is a
one-line edit there.
