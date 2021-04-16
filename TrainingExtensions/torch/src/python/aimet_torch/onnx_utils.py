# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2017-2020, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

# =============================================================================
#  @@-COPYRIGHT-START-@@
# From PyTorch:
#
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
# From Caffe2:
#
# Copyright (c) 2016-present, Facebook Inc. All rights reserved.
#
# All contributions by Facebook:
# Copyright (c) 2016 Facebook Inc.
#
# All contributions by Google:
# Copyright (c) 2015 Google Inc.
# All rights reserved.
#
# All contributions by Yangqing Jia:
# Copyright (c) 2015 Yangqing Jia
# All rights reserved.
#
# All contributions by Kakao Brain:
# Copyright 2019-2020 Kakao Brain
#
# All contributions from Caffe:
# Copyright(c) 2013, 2014, 2015, the respective contributors
# All rights reserved.
#
# All other contributions:
# Copyright(c) 2015, 2016 the respective contributors
# All rights reserved.
#
# Caffe2 uses a copyright model similar to Caffe: each contributor holds
# copyright over their contributions to Caffe2. The project versioning records
# all such contribution and copyright details. If a contributor wants to further
# mark their specific copyright on a particular contribution, they should
# indicate their copyright solely in the commit message of the change when it is
# committed.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
#    and IDIAP Research Institute nor the names of its contributors may be
#    used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#  @@-COPYRIGHT-END-@@
# =============================================================================


""" Utilities to load and save onnx models """

from typing import Union, List, Tuple, Dict
import os
import copy
import importlib
from inspect import getmembers, isfunction

import torch
import torch.nn as nn
import torch.onnx.symbolic_caffe2

import torch.onnx.symbolic_registry as sym_registry

import onnx

from aimet_common.utils import AimetLogger
import aimet_torch.utils
import aimet_torch.elementwise_ops as elementwise_ops
from aimet_torch.defs import PassThroughOp, OpToIOTensors

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)


# This is a dict that maps a PyTorch module type to the corresponding ONNX op type (as a string)
map_torch_types_to_onnx = {
    nn.Conv2d: ['Conv'],
    nn.Dropout: ['Dropout'],
    nn.Dropout2d: ['Dropout'],
    nn.BatchNorm1d: ['BatchNormalization'],
    nn.BatchNorm2d: ['BatchNormalization'],
    nn.ReLU: ['Relu'],
    nn.ReLU6: ['Clip'],
    nn.MaxPool2d: ['MaxPool'],
    nn.Linear: ['Gemm', 'MatMul'],
    nn.AdaptiveAvgPool2d: ['GlobalAveragePool', 'AveragePool'],
    nn.AvgPool2d: ['AveragePool'],
    nn.LogSoftmax: ['LogSoftmax'],
    nn.RNN:  ['RNN'],
    nn.LSTM: ['LSTM'],
    nn.GRU: ['GRU'],
    nn.ConvTranspose2d: ['ConvTranspose'],
    nn.Sigmoid: ['Sigmoid'],
    nn.Upsample: ['Upsample'],
    nn.PReLU: ['PRelu'],
    nn.LeakyReLU: ['LeakyRelu'],
    nn.Flatten: ['Flatten'],
    elementwise_ops.Add: ['Add'],
    elementwise_ops.Subtract: ['Sub'],
    elementwise_ops.Multiply: ['Mul'],
    elementwise_ops.Divide: ['Div'],
    elementwise_ops.Concat: ['Concat']

}

# Define this as a list instead of tuple to allow for users to modify
torch_types_to_ignore = [nn.Dropout, nn.Dropout2d, nn.Identity, PassThroughOp]
torch_recurrent_modules = (nn.RNN, nn.LSTM, nn.GRU)

# Maps pytorch functional op string names to corresponding onnx types.
pytorch_functional_name_to_onnx_dict = {
    'add': 'Add',
    'cat': 'Concat',
    'mul': 'Mul',
    'div': 'Div'
}


def register_quantized_ops(domain, version):
    """
    Copied and modified code for torch/onnx/symbolic_caffe2.py
    This is a temporary measure - Calling this function as-is was causing unexplained memory errors later on
    This can perhaps be replaced by registering the marker layers as custom ONNX ops
    """
    # pylint: disable=protected-access

    # Register all the non-quantized ops
    sym_registry.register_version('', version)
    # Register all quantized ops
    module = importlib.import_module('torch.onnx.symbolic_caffe2')
    sym_registry._symbolic_versions['caffe2'] = module
    quant_version_ops = getmembers(sym_registry._symbolic_versions['caffe2'])
    for op in quant_version_ops:
        if isfunction(op[1]) and not sym_registry.is_registered_op(op[0], domain, version):
            aten_q_ops = ['dequantize', 'quantize_per_tensor']
            if op[0] in aten_q_ops:
                sym_registry.register_op(op[0], op[1], '', version)


register_quantized_ops('caffe2', 9)


class OnnxSaver:
    """
    Utilities to save/load onnx models
    """

    @classmethod
    def set_node_names(cls, onnx_model_path: str, pytorch_model: torch.nn.Module,
                       dummy_input: Union[torch.Tensor, Tuple]):
        """
        This utility loads a given onnx model file and set the names of all the nodes (ops) to equivalent
        pytorch module names given the corresponding pytorch model.
        :param onnx_model_path: Path to the ONNX model file
        :param pytorch_model: Equivalent PyTorch model instance
        :param dummy_input: Dummy input to the model. Used to parse model graph.
        :return:
        """

        # Load the model
        onnx_model = onnx.load(onnx_model_path)

        # Parse the ONNX model and create mapping from input and output tensors to corresponding nodes
        map_output_tensor_to_node, _ = cls.create_map_of_tensor_to_node(onnx_model)

        ordered_list_of_nodes = cls.find_ordered_list_of_onnx_nodes(map_output_tensor_to_node)

        ordered_list_of_nodes = cls._filter_out_uninteresting_onnx_nodes(pytorch_model, dummy_input,
                                                                         ordered_list_of_nodes,
                                                                         os.path.dirname(onnx_model_path))

        # Find corresponding pytorch nodes for every ONNX node
        # and set the name of the ONNX nodes to the names of the corresponding PyTorch modules
        cls.map_onnx_nodes_to_pytorch(pytorch_model, dummy_input, ordered_list_of_nodes)

        # Save back the onnx model file
        onnx.save(onnx_model, onnx_model_path)

    @staticmethod
    def create_map_of_tensor_to_node(onnx_model: onnx.ModelProto) -> Tuple[Dict[str, List[onnx.NodeProto]],
                                                                           Dict[str, onnx.NodeProto]]:
        """
        Create and return two dicts
            1. Tensor -> list of nodes that consume this tensor
            2. Tensor -> node that produces this tensor
        :param onnx_model: ONNX model object
        :return: The two dicts described above

        Note: The list in #1 is ordered exactly in the order that pytorch trace reaches these nodes. This is important
        because later on we will use pytorch layer hooks to match these nodes with the equivalent PyTorch modules.
        The expectation is that PyTorch trace and PyTorch hooks follow the same execution sequence
        """
        map_input_tensor_to_node = {}
        map_output_tensor_to_node = {}
        for node in onnx_model.graph.node:
            for in_tensor in node.input:
                if in_tensor in map_input_tensor_to_node:
                    map_input_tensor_to_node[in_tensor].append(node)
                else:
                    map_input_tensor_to_node[in_tensor] = [node]

            for output in node.output:
                assert output not in map_output_tensor_to_node, 'More than one node produces the same tensor'
                map_output_tensor_to_node[output] = node

        return map_output_tensor_to_node, map_input_tensor_to_node

    @staticmethod
    def find_ordered_list_of_onnx_nodes(map_output_tensor_to_node: Dict[str, onnx.NodeProto]) -> List[onnx.NodeProto]:
        """
        Given a ONNX model find all the nodes that are inputs to the model - meaning nodes who have no
        preceeding nodes
        :param map_output_tensor_to_node: Map of tensor->node that produces this tensor
        :return: List of input nodes
        """
        node_output_tensor_pairs = [(node, output) for output, node in map_output_tensor_to_node.items()]

        # Filter out dead outputs
        node_output_tensor_pairs = [(node, output) for output, node in map_output_tensor_to_node.items()
                                    if output.isnumeric()]

        node_output_tensor_pairs.sort(key=lambda x: int(x[1]))
        ordered_nodes = [node for node, _ in node_output_tensor_pairs]

        # Remove duplicates - multiple output nodes will be duplicates in the above list
        ordered_nodes_no_duplicates = []
        set_of_ordered_nodes = set()
        for node in ordered_nodes:
            if id(node) not in set_of_ordered_nodes:
                set_of_ordered_nodes.add(id(node))
                ordered_nodes_no_duplicates.append(node)

        return ordered_nodes_no_duplicates

    @classmethod
    def _add_markers(cls, starting_module):
        """Recursively add marker layers
        """

        class MarkerLayer(torch.nn.Module):
            """
            This is a temporary layer that in inserted next to a real layer to distinguish the real layer in the
            exported ONNX format
            """
            def __init__(self, module):
                super(MarkerLayer, self).__init__()
                self.module = module

            def forward(self, *x):
                """
                Forward pass for Marker layer
                """
                o = []
                for i, t in enumerate(x):
                    if i == 0:
                        t = torch.quantize_per_tensor(t, 1, 0, torch.qint32)
                        t = torch.dequantize(t)
                    o.append(t)

                x = self.module(*o)
                if isinstance(x, torch.Tensor):
                    x = torch.quantize_per_tensor(x, 2, 0, torch.qint32)
                    x = torch.dequantize(x)
                else:
                    o = []
                    for i, t in enumerate(x):
                        if i == 0:
                            t = torch.quantize_per_tensor(t, 1, 0, torch.qint32)
                            t = torch.dequantize(t)
                        o.append(t)
                    x = tuple(o)

                return x

        for module_name, module_ref in starting_module.named_children():

            if aimet_torch.utils.is_leaf_module(module_ref):
                marker_layer = MarkerLayer(module_ref)
                setattr(starting_module, module_name, marker_layer)

            # recursively call children modules
            else:
                cls._add_markers(module_ref)

    @classmethod
    def _filter_out_uninteresting_onnx_nodes(cls, pt_model, dummy_input, onnx_ordered_node_list, working_dir):

        model = copy.deepcopy(pt_model).cpu()
        cls._add_markers(model)

        temp_file = os.path.join(working_dir, 'temp_onnx_model_with_markers.onnx')
        torch.onnx.export(model, dummy_input, temp_file, training=torch.onnx.TrainingMode.TRAINING)

        # Parse the ONNX model and create mapping from input and output tensors to corresponding nodes
        map_output_tensor_to_node_marker_model, _ = cls.create_map_of_tensor_to_node(onnx.load(temp_file))
        ordered_list_of_nodes_marker_model = cls.find_ordered_list_of_onnx_nodes(
            map_output_tensor_to_node_marker_model)

        skip_info = []
        node_iter = iter(ordered_list_of_nodes_marker_model)
        while node_iter:

            # Find the next block of interest
            node = next(node_iter, None)
            if node is None:
                break
            while node.op_type != 'Int8Quantize':
                skip_info.append((node.op_type, True))
                node = next(node_iter, None)
                if node is None:
                    break
            if node is None:
                break

            # Skip the marker nodes
            node = next(node_iter)
            assert node.op_type == 'Int8Dequantize'

            # Glob the block of nodes (these are the nodes corresponding to the pytorch modules)
            node = next(node_iter)
            while node.op_type != 'Int8Quantize':
                skip_info.append((node.op_type, False))
                node = next(node_iter)

            # Skip the marker nodes
            node = next(node_iter)
            assert node.op_type == 'Int8Dequantize'

        # Use the skip list to filter out the original list of onnx nodes
        assert len(skip_info) == len(onnx_ordered_node_list)
        filtered_node_list = []
        for index, node in enumerate(onnx_ordered_node_list):
            assert node.op_type == skip_info[index][0]
            if not skip_info[index][1]:
                filtered_node_list.append(node)

        return filtered_node_list

    @staticmethod
    def get_num_onnx_nodes_to_map(module: nn.Module):
        """
        Get the number of onnx nodes that map to the same torch module
        :param module: PyTorch model instance
        :return: number of onnx nodes:
        """
        if isinstance(module, torch_recurrent_modules):
            return module.num_layers
        return 1

    @staticmethod
    def map_onnx_nodes_to_pytorch(torch_model: nn.Module, dummy_input: Union[torch.Tensor, Tuple],
                                  onnx_ordered_list: List[onnx.NodeProto]):
        """
        Find the ONNX node that corresponds to each PyTorch module. And sets the name of the ONNX mode to that of the
        PyTorch module
        :param torch_model: PyTorch model instance
        :param dummy_input: Dummy input to the model. Used to parse model graph.
        :param onnx_ordered_list: An ordinally-ordered list of ONNX nodes
        :return:
        """
        torch_ordered_list = aimet_torch.utils.get_ordered_list_of_modules(torch_model, dummy_input)

        torch_index = 0
        onnx_index = 0

        num_onnx_nodes_to_map_to_same_torch_node = 0
        while torch_index < len(torch_ordered_list):
            # If few PyTorch ops are not mapped to ONNX ops
            if onnx_index >= len(onnx_ordered_list):
                _logger.warning('All ONNX ops were exhausted but few PyTorch ops did not get mapped to a '
                                'corresponding ONNX op')
                break
            name, module = torch_ordered_list[torch_index]

            if isinstance(module, tuple(torch_types_to_ignore)):
                torch_index += 1
                continue

            if onnx_ordered_list[onnx_index].op_type in map_torch_types_to_onnx[type(module)]:
                _logger.debug('Found a match: %r -> %r', onnx_ordered_list[onnx_index].op_type, name)
                onnx_ordered_list[onnx_index].name = name

                if num_onnx_nodes_to_map_to_same_torch_node == 0:
                    num_onnx_nodes_to_map_to_same_torch_node = OnnxSaver.get_num_onnx_nodes_to_map(module)

                num_onnx_nodes_to_map_to_same_torch_node = num_onnx_nodes_to_map_to_same_torch_node - 1
                if num_onnx_nodes_to_map_to_same_torch_node == 0:
                    torch_index += 1

            onnx_index += 1

    @classmethod
    def find_model_input_nodes(cls, onnx_model: onnx.ModelProto,
                               map_output_tensor_to_node: Dict[str, onnx.NodeProto]) -> List[onnx.NodeProto]:
        """
        Given a ONNX model find all the nodes that are inputs to the model - meaning nodes who have no
        preceeding nodes
        :param onnx_model: ONNX model instance
        :param map_output_tensor_to_node: Map of tensor->node that produces this tensor
        :return: List of input nodes
        """
        input_nodes = []
        for node in onnx_model.graph.node:
            preceeding_nodes = cls.find_preceeding_nodes(node, map_output_tensor_to_node)
            if not preceeding_nodes:
                input_nodes.append(node)

        return input_nodes

    @staticmethod
    def find_preceeding_nodes(node: onnx.NodeProto,
                              map_output_tensor_to_node: Dict[str, onnx.NodeProto]) -> List[onnx.NodeProto]:
        """
        Given an ONNX node, find the nodes that directly feed into this node
        :param node: ONNX node
        :param map_output_tensor_to_node: Map of tensor->node that produces this tensor
        :return:
        """

        nodes = []
        for in_tensor in node.input:
            if in_tensor in map_output_tensor_to_node:
                nodes.append(map_output_tensor_to_node[in_tensor])

        return nodes

    @staticmethod
    def is_onnx_tensor_valid_param(onnx_tensor_name: str):
        """
        This is based on the assumption that parameters have string names and not just numeric.
        param names are non-numeric like "classifier.weight" etc, and not only numeric
        ONNX tensor names that is used for tensors such as "123", "14" etc.
        :param onnx_tensor_name: Name of the ONNX tensor
        :return: True, if valid param, False otherwise.
        """

        if onnx_tensor_name.isnumeric():
            return False

        return True

    @staticmethod
    def get_onnx_node_to_io_tensor_names_map(onnx_model: onnx.NodeProto) -> \
            (Dict[str, Union[OpToIOTensors, List[OpToIOTensors]]], set):
        """
        Given an ONNX model, gets the inputs and output tensor names for each node in the model.
        if multiple onnx nodes have the same name then the nodes are provided as a list of inputs and output tensor
         names, one for each onnx node.
        :param onnx_model: The ONNX model instance
        :return: Dictionary of ONNX node name and corresponding input and output tensor names and a set with all valid
        param names in model
        """

        node_to_io_tensor_name_map = {}
        valid_param_set = set()

        for node in onnx_model.graph.node:
            if node.name:
                onnx_node_io_tensors = OpToIOTensors(list(node.input), list(node.output))
                if node.name not in node_to_io_tensor_name_map:
                    node_to_io_tensor_name_map[node.name] = onnx_node_io_tensors
                else:
                    # get the torch module associate with the onnx node
                    torch_module = [module for module, onnx_nodes in map_torch_types_to_onnx.items()
                                    if node.op_type in onnx_nodes]
                    assert len(torch_module) == 1
                    # Check if the torch module corresponding to tonnx node generates many to one mapping
                    if torch_module[0] not in torch_recurrent_modules:
                        # onnx module is being reused in the model pick the last entry
                        node_to_io_tensor_name_map[node.name] = onnx_node_io_tensors
                        continue

                    # if an entry with a single IOTensors exists then convert the entry to a list
                    if not isinstance(node_to_io_tensor_name_map[node.name], list):
                        node_to_io_tensor_name_map[node.name] = [node_to_io_tensor_name_map[node.name],
                                                                 onnx_node_io_tensors]
                    else:
                        node_to_io_tensor_name_map[node.name].append(onnx_node_io_tensors)

            # update valid params list
            for input_tensor in list(node.input):
                if OnnxSaver.is_onnx_tensor_valid_param(input_tensor):
                    valid_param_set.add(input_tensor)

        return node_to_io_tensor_name_map, valid_param_set
