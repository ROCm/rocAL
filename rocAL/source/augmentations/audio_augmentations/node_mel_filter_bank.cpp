/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "node_mel_filter_bank.h"

#include <vx_ext_rpp.h>

#include "exception.h"

MelFilterBankNode::MelFilterBankNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) : Node(inputs, outputs) {}

void MelFilterBankNode::create_node() {
    if (_node)
        return;

    vx_scalar freq_high = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_freq_high);
    vx_scalar freq_low = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_freq_low);
    vx_scalar mel_formula = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, &_mel_formula);
    vx_scalar nfilter = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &_nfilter);
    vx_scalar normalize = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_BOOL, &_normalize);
    vx_scalar sample_rate = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, &_sample_rate);
    _node = vxExtRppMelFilterBank(_graph->get(), _inputs[0]->handle(), _inputs[0]->get_roi_tensor(), _outputs[0]->handle(), _outputs[0]->get_roi_tensor(), freq_high,
                                  freq_low, mel_formula, nfilter, normalize, sample_rate);

    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the mel filter bank (vxRppMelFilterBank) node failed: " + TOSTR(status))
}

void MelFilterBankNode::update_node() {}

void MelFilterBankNode::init(float freq_high, float freq_low, RocalMelScaleFormula mel_formula,
                             int nfilter, bool normalize, float sample_rate) {
    _freq_high = freq_high;
    _freq_low = freq_low;
    _mel_formula = mel_formula;
    _nfilter = nfilter;
    _normalize = normalize;
    _sample_rate = sample_rate;
}
