# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX-ir related utility functions"""

from typing import Optional

import numpy as np
import onnx_ir
import onnx
from aimet_onnx.graph_passes.fusions.fusion_registry import AIMET_SUPERGROUP_DOMAIN

#: ONNX QDQ node types. Unlike ``QcQuantizeOp`` these come in pairs, so removing
#: them takes a use-redirect per node rather than a single output->input map.
_ONNX_QDQ_TYPES = frozenset(("QuantizeLinear", "DequantizeLinear"))


def static_tensor(value: Optional[onnx_ir.Value]) -> Optional[onnx_ir.TensorProtocol]:
    """Return the constant tensor behind ``value``, or None if it is dynamic.

    Covers both forms ConnectedGraph reports as ``is_parm``/``is_const``: a graph
    initializer, and the output of a ``Constant`` node (which is what
    ``do_constant_folding`` exports emit for e.g. RMSNorm gammas).
    """
    if value is None:
        return None
    return onnx_ir.convenience.get_const_tensor(value)


def is_static(value: Optional[onnx_ir.Value]) -> bool:
    """Return True if ``value`` is an initializer or a ``Constant`` node output."""
    return static_tensor(value) is not None


def set_static_tensor(value: onnx_ir.Value, array: np.ndarray) -> None:
    """Overwrite the constant tensor behind ``value`` with ``array``, in place.

    The write-side mirror of :func:`static_tensor`: it covers the same two forms,
    an initializer and the output of a ``Constant`` node, so a caller that read a
    weight through ``static_tensor`` can write it back without caring which form
    holds it.

    Shape and dtype must be preserved. A transform that changes either has
    rewritten the tensor's contract with every consumer (and with the graph's
    ``value_info``), which cannot be expressed by swapping one tensor.

    :param value: The static Value to overwrite.
    :param array: Replacement data, same shape and dtype as the current tensor.
    :raises ValueError: If ``value`` is not static, if shape or dtype differ, or
        if the producing ``Constant`` node carries its data in an attribute form
        other than ``value`` (e.g. ``value_floats``).
    """
    current = static_tensor(value)
    if current is None:
        raise ValueError(
            f"Value '{value.name}' is not static (no initializer and no Constant "
            "producer), so it has no constant tensor to overwrite."
        )

    replacement = onnx_ir.tensor(array, name=current.name)
    if tuple(replacement.shape) != tuple(current.shape):
        raise ValueError(
            f"Value '{value.name}': replacement shape {tuple(replacement.shape)} "
            f"differs from the current shape {tuple(current.shape)}."
        )
    if replacement.dtype != current.dtype:
        raise ValueError(
            f"Value '{value.name}': replacement dtype {replacement.dtype} differs "
            f"from the current dtype {current.dtype}. Cast before writing back."
        )

    if value.const_value is not None:
        value.const_value = replacement
        return

    # Constant node: the data lives in the node's attribute, not on the Value.
    # get_const_tensor accepts several attribute spellings, but only ``value``
    # holds a tensor; the others are scalar/list forms that a weight never uses.
    node = value.producer()
    attr_name = next(iter(node.attributes))
    if attr_name != "value":
        raise ValueError(
            f"Constant node '{node.name}' holds its data in attribute "
            f"'{attr_name}'; only the 'value' (tensor) form can be overwritten."
        )
    node.attributes["value"] = onnx_ir.AttrTensor("value", replacement)


def remove_quantizers(model: onnx_ir.Model) -> None:
    """Remove every quantizer node, rewiring consumers back to the source tensor.

    Covers AIMET's ``QcQuantizeOp`` (via :func:`remove_aimet_quantizers`) and ONNX
    ``QuantizeLinear``/``DequantizeLinear`` pairs, so a quantized graph presents
    the same topology — and the same tensor names — as the float graph it was
    built from.

    :param model: Model to strip, mutated in place.
    """
    remove_aimet_quantizers(model)
    _remove_onnx_qdq(model)


def _remove_onnx_qdq(model: onnx_ir.Model) -> None:
    """Collapse ``QuantizeLinear``/``DequantizeLinear`` pass-throughs in place."""
    _remove_passthrough_nodes(
        [node for node in model.graph if node.op_type in _ONNX_QDQ_TYPES]
    )


def remove_aimet_quantizers(model: onnx_ir.Model):
    """Remove AIMET ``QcQuantizeOp`` nodes, rewiring consumers to the source tensor."""
    _remove_passthrough_nodes(
        [node for node in model.graph.all_nodes() if node.op_type == "QcQuantizeOp"]
    )


def _remove_passthrough_nodes(nodes: list) -> None:
    """Delete single-in/single-out ``nodes``, redirecting their uses to input 0.

    A graph output produced by a pass-through must be re-pointed at the source
    ``Value``, not merely renamed to match it. Renaming leaves ``graph.outputs``
    holding the deleted node's Value, which makes the real producer's output an
    unused, non-output tensor — so the whole graph reads as dead code to any
    later IR pass. It survives an immediate ``to_proto`` (which matches tensors
    by name) and nothing else.

    Because the source Value may carry no declared type or shape while the
    pass-through's output does, the annotation is copied across before the
    re-point; ``onnx.checker`` requires a type on every graph output.
    """
    passthroughs = [
        node
        for node in nodes
        if len(node.outputs) == 1 and node.inputs and node.inputs[0] is not None
    ]

    for node in passthroughs:
        source, produced = node.inputs[0], node.outputs[0]
        if produced.is_graph_output():
            if source.type is None:
                source.type = produced.type
            if source.shape is None:
                source.shape = produced.shape
        # Nodes are visited in graph order, so a chain (QuantizeLinear ->
        # DequantizeLinear) has already had its head redirected and `source` is
        # the true origin by the time the tail is processed.
        onnx_ir.convenience.replace_all_uses_with(
            produced, source, replace_graph_outputs=True
        )

    for node in passthroughs:
        # safe=True detaches the node from its inputs' user lists; without it the
        # node keeps counting as a consumer after removal. Remove from the node's
        # own graph so nodes inside a subgraph are handled.
        node.graph.remove(node, safe=True)


def get_constant_singleton_value(
    value: onnx_ir.Value | onnx_ir.Attr | None,
) -> float | None:
    """Get the constant singleton value from an ONNX IR Value, if it exists.

    Args:
        value: The ONNX IR Value to extract the constant from.
    Returns:
        The constant singleton value as a float, or None if not found.
    """
    numpy_value = get_constant_or_attribute_value(value)

    if numpy_value is None or numpy_value.size != 1:
        return None

    return numpy_value.flatten()[0].item()


def get_constant_as_array(value: onnx_ir.Value | None) -> np.ndarray | None:
    """Get the constant singleton value from an ONNX IR Value, if it exists.

    Args:
        value: The ONNX IR Value to extract the constant from.
    Returns:
        The constant singleton value as a float, or None if not found.
    """
    const_value = static_tensor(value)
    if const_value is None:
        return None

    return const_value.numpy()


def get_constant_or_attribute_value(
    value: onnx_ir.Value | onnx_ir.Attr | None,
) -> None | np.ndarray:
    """Get the constant value from an ONNX IR Value or Attr, if it exists."""
    if value is None:
        return None
    if isinstance(value, onnx_ir.Value):
        return get_constant_as_array(value)
    if isinstance(value, onnx_ir.Attr):
        return np.asarray(value.value)
    raise RuntimeError(f"Received unexpected type for value: {type(value)}")


def _sort_functions_hierarchically(model: onnx_ir.Model) -> None:
    """Sort model functions from outermost to innermost to prevent mangling of names during inlining."""
    # pylint: disable=protected-access
    sorted_funcs = {}

    def node_has_impl(node: onnx_ir.Node) -> bool:
        return not is_fused_supergroup(node) or node.op_identifier() in sorted_funcs

    while True:
        runnable_functions = {
            fid: func
            for fid, func in model.functions.items()
            if fid not in sorted_funcs
            and all(node_has_impl(n) for n in func.graph.all_nodes())
        }
        if not runnable_functions:
            break
        sorted_funcs.update(runnable_functions)

    if not sorted_funcs.keys() == model.functions.keys():
        raise RuntimeError(
            f"Cycle detected among supergroup functions: {set(model.functions.keys()) - set(sorted_funcs.keys())}"
        )

    # Reverse ordering to prevent name mangling while unrolling
    model._functions = dict(reversed(list(sorted_funcs.items())))


def inline_all_supergroups(model: onnx_ir.Model) -> None:
    """Inline all aimet supergroup functions, restoring original node and value names."""
    supergroup_functions = {
        func for func in model.functions.values() if is_fused_supergroup(func)
    }
    if not supergroup_functions:
        return

    _sort_functions_hierarchically(model)
    onnx_ir.passes.common.InlinePass(lambda f: f in supergroup_functions).call(model)
    supergroup_opsets = [
        opset
        for opset in model.graph.opset_imports
        if opset.startswith(AIMET_SUPERGROUP_DOMAIN)
    ]
    for name in supergroup_opsets:
        model.graph.opset_imports.pop(name)


def unique_name(base: str, existing: set[str]) -> str:
    """Generate a unique name based on the provided base that does not exist in the existing set."""
    if base not in existing:
        return base
    i = 1
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def get_upstream_cast_type(value: onnx_ir.Value) -> int | None:
    """Return the ``to`` attribute of an upstream Cast producer, if any"""
    producer = value.producer()
    if producer is None or producer.op_type != "Cast" or producer.domain != "":
        return None
    to_attr = producer.attributes.get("to")
    return to_attr.as_int() if to_attr is not None else None


def fold_overload_into_op_domain(model: onnx_ir.Model):
    """
    Works around bug in onnxruntime dispatch to overloaded functions by replacing
    function.domain with "{function.domain}.{function.overload}" for all supergroup ops
    """
    new_functions = {}
    for function in model.functions.values():
        if function.domain == AIMET_SUPERGROUP_DOMAIN and function.overload:
            function.domain = f"{function.domain}.{function.overload}"
        new_functions[function.identifier()] = function

    for node in model.graph.all_nodes():
        if node.domain != AIMET_SUPERGROUP_DOMAIN:
            continue
        if not node.overload:
            continue
        node.domain = f"{node.domain}.{node.overload}"

    model._functions = new_functions  # pylint:disable = protected-access
    opsets = set(f.domain for f in model.functions.values() if is_fused_supergroup(f))
    for opset in opsets:
        model.opset_imports[opset] = 1


def is_fused_supergroup(
    node: onnx_ir.Node | onnx_ir.Function | onnx.NodeProto | onnx.FunctionProto,
) -> bool:
    """Return True if ``node`` represents an AIMET supergroup op."""
    return node.domain.startswith(AIMET_SUPERGROUP_DOMAIN)
