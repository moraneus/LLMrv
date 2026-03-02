"""
Policies and propositions API router.

CRUD for propositions and policies with formula validation.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.engine.ptltl import ParseError, PropNode, parse
from backend.models.policy import Policy, Proposition
from backend.routers.chat import invalidate_monitors
from backend.store.db import DatabaseStore

router = APIRouter(tags=["policies"])


def _get_db(request: Request) -> DatabaseStore:
    return request.app.state.db


# Request schemas


class CreatePropositionRequest(BaseModel):
    """Request body for creating a proposition."""

    prop_id: str
    description: str
    role: str  # "user" | "assistant"


class UpdatePropositionRequest(BaseModel):
    """Request body for updating a proposition."""

    description: str | None = None
    role: str | None = None


class CreatePolicyRequest(BaseModel):
    """Request body for creating a policy."""

    name: str
    formula_str: str
    enabled: bool = True


class UpdatePolicyRequest(BaseModel):
    """Request body for updating a policy."""

    name: str | None = None
    formula_str: str | None = None
    enabled: bool | None = None


# Helper: extract prop_ids from AST


def _extract_prop_ids(node) -> set[str]:
    """Recursively extract all proposition IDs from a parsed AST."""
    if isinstance(node, PropNode):
        return {node.prop_id}
    result: set[str] = set()
    for field_name in ("child", "left", "right"):
        child = getattr(node, field_name, None)
        if child is not None:
            result |= _extract_prop_ids(child)
    return result


async def _validate_formula(db: DatabaseStore, formula_str: str) -> tuple[list[str], str | None]:
    """Parse formula, validate prop references. Returns (prop_ids, error_or_none)."""
    try:
        ast = parse(formula_str)
    except ParseError as e:
        return [], str(e)

    prop_ids = sorted(_extract_prop_ids(ast))

    # Check all referenced propositions exist
    missing = []
    for pid in prop_ids:
        prop = await db.get_proposition(pid)
        if prop is None:
            missing.append(pid)

    if missing:
        return prop_ids, f"Unknown propositions: {', '.join(missing)}"

    return prop_ids, None


# Propositions endpoints


@router.get("/propositions")
async def list_propositions(request: Request) -> list[Proposition]:
    """List all propositions."""
    db = _get_db(request)
    rows = await db.list_propositions()
    return [
        Proposition(
            prop_id=r["prop_id"],
            description=r["description"],
            role=r["role"],
        )
        for r in rows
    ]


@router.post("/propositions", status_code=201)
async def create_proposition(request: Request, body: CreatePropositionRequest) -> Proposition:
    """Create a new proposition."""
    db = _get_db(request)

    # Validate prop_id is not empty
    if not body.prop_id or not body.prop_id.strip():
        raise HTTPException(422, "Proposition ID cannot be empty.")

    # Validate description is not empty
    if not body.description or not body.description.strip():
        raise HTTPException(422, "Proposition description cannot be empty.")

    if body.role not in ("user", "assistant"):
        raise HTTPException(422, f"Invalid role: {body.role}. Must be 'user' or 'assistant'.")

    existing = await db.get_proposition(body.prop_id)
    if existing:
        raise HTTPException(409, f"Proposition '{body.prop_id}' already exists.")

    await db.create_proposition(body.prop_id, body.description, body.role)
    invalidate_monitors()
    return Proposition(prop_id=body.prop_id, description=body.description, role=body.role)


@router.put("/propositions/{prop_id}")
async def update_proposition(
    request: Request, prop_id: str, body: UpdatePropositionRequest
) -> Proposition:
    """Update an existing proposition."""
    db = _get_db(request)
    existing = await db.get_proposition(prop_id)
    if not existing:
        raise HTTPException(404, f"Proposition '{prop_id}' not found.")

    if body.role is not None and body.role not in ("user", "assistant"):
        raise HTTPException(422, f"Invalid role: {body.role}. Must be 'user' or 'assistant'.")

    await db.update_proposition(prop_id, description=body.description, role=body.role)
    invalidate_monitors()
    updated = await db.get_proposition(prop_id)
    return Proposition(
        prop_id=updated["prop_id"],
        description=updated["description"],
        role=updated["role"],
    )


@router.delete("/propositions/{prop_id}", status_code=204)
async def delete_proposition(request: Request, prop_id: str):
    """Delete a proposition. Rejects if referenced by any policy."""
    db = _get_db(request)
    existing = await db.get_proposition(prop_id)
    if not existing:
        raise HTTPException(404, f"Proposition '{prop_id}' not found.")

    # Check for referencing policies
    referencing = await db.get_policies_using_proposition(prop_id)
    if referencing:
        names = [r["name"] for r in referencing]
        raise HTTPException(
            409,
            f"Cannot delete proposition '{prop_id}': referenced by "
            f"policies: {', '.join(names)}. Remove it from those policies first.",
        )

    await db.delete_proposition(prop_id)
    invalidate_monitors()


# Policies endpoints


@router.get("/policies")
async def list_policies(request: Request) -> list[Policy]:
    """List all policies with their proposition references."""
    db = _get_db(request)
    rows = await db.list_policies()
    result = []
    for r in rows:
        props = await db.get_policy_propositions(r["policy_id"])
        result.append(
            Policy(
                policy_id=r["policy_id"],
                name=r["name"],
                formula_str=r["formula_str"],
                propositions=props,
                enabled=bool(r["enabled"]),
            )
        )
    return result


MAX_FORMULA_LENGTH = 1000
MAX_POLICY_COUNT = 50


@router.post("/policies", status_code=201)
async def create_policy(request: Request, body: CreatePolicyRequest) -> Policy:
    """Create a new policy. Validates the ptLTL formula and proposition references."""
    db = _get_db(request)

    # Validate name is not empty
    if not body.name or not body.name.strip():
        raise HTTPException(422, "Policy name cannot be empty.")

    # Validate formula is not empty
    if not body.formula_str or not body.formula_str.strip():
        raise HTTPException(422, "Formula cannot be empty.")

    # Validate formula size
    if len(body.formula_str) > MAX_FORMULA_LENGTH:
        raise HTTPException(422, f"Formula too long. Maximum {MAX_FORMULA_LENGTH} characters.")

    # Validate policy count limit
    existing_policies = await db.list_policies()
    if len(existing_policies) >= MAX_POLICY_COUNT:
        raise HTTPException(422, f"Maximum of {MAX_POLICY_COUNT} policies reached.")

    prop_ids, error = await _validate_formula(db, body.formula_str)
    if error:
        raise HTTPException(422, error)

    policy_id = str(uuid.uuid4())
    await db.create_policy(policy_id, body.name, body.formula_str, body.enabled)
    await db.set_policy_propositions(policy_id, prop_ids)
    invalidate_monitors()

    return Policy(
        policy_id=policy_id,
        name=body.name,
        formula_str=body.formula_str,
        propositions=prop_ids,
        enabled=body.enabled,
    )


@router.put("/policies/{policy_id}")
async def update_policy(request: Request, policy_id: str, body: UpdatePolicyRequest) -> Policy:
    """Update an existing policy. Re-validates formula if changed."""
    db = _get_db(request)
    existing = await db.get_policy(policy_id)
    if not existing:
        raise HTTPException(404, f"Policy '{policy_id}' not found.")

    # If formula changed, re-validate
    if body.formula_str is not None:
        if len(body.formula_str) > MAX_FORMULA_LENGTH:
            raise HTTPException(422, f"Formula too long. Maximum {MAX_FORMULA_LENGTH} characters.")
        prop_ids, error = await _validate_formula(db, body.formula_str)
        if error:
            raise HTTPException(422, error)
        await db.set_policy_propositions(policy_id, prop_ids)

    await db.update_policy(
        policy_id,
        name=body.name,
        formula_str=body.formula_str,
        enabled=body.enabled,
    )
    invalidate_monitors()

    updated = await db.get_policy(policy_id)
    props = await db.get_policy_propositions(policy_id)
    return Policy(
        policy_id=updated["policy_id"],
        name=updated["name"],
        formula_str=updated["formula_str"],
        propositions=props,
        enabled=bool(updated["enabled"]),
    )


@router.delete("/policies/{policy_id}", status_code=204)
async def delete_policy(request: Request, policy_id: str):
    """Delete a policy."""
    db = _get_db(request)
    existing = await db.get_policy(policy_id)
    if not existing:
        raise HTTPException(404, f"Policy '{policy_id}' not found.")
    await db.delete_policy(policy_id)
    invalidate_monitors()


@router.post("/policies/validate")
async def validate_formula(request: Request, body: CreatePolicyRequest):
    """Validate a ptLTL formula without creating a policy."""
    db = _get_db(request)
    prop_ids, error = await _validate_formula(db, body.formula_str)
    if error:
        return {"valid": False, "error": error, "propositions": prop_ids}
    return {"valid": True, "error": None, "propositions": prop_ids}
