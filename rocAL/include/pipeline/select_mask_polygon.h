/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once
#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "pipeline/tensor.h"

/*! \brief Selects polygons from mask metadata by mask ID.
 *
 * Given per-sample polygon masks (from COCO-style segmentation metadata),
 * filters the polygons to only include those whose object index appears in
 * the requested mask_ids list.
 */
class SelectMaskPolygon {
   public:
    SelectMaskPolygon(size_t user_batch_size);

    /*! \brief Run polygon selection on a batch.
     *  \param [in]  mask_data          Batch of polygon coordinate tensors.
     *  \param [in]  polygon_counts     Per-sample per-object polygon counts.
     *  \param [in]  vertices_counts    Per-sample per-object per-polygon vertex counts.
     *  \param [in]  mask_ids           Object indices to select.
     *  \param [out] sel_vertices_counts Per-sample selected vertex counts.
     *  \param [out] sel_mask_ids       Per-sample selected mask IDs.
     *  \param [in]  reindex_mask       If true, remap mask IDs to sequential indices.
     *  \param [out] output_list        TensorList to populate.
     *  \return Pointer to output_list after population.
     */
    TensorList *run(rocalTensorList *mask_data,
                    std::vector<std::vector<int>> polygon_counts,
                    std::vector<std::vector<std::vector<int>>> vertices_counts,
                    std::vector<int> mask_ids,
                    std::vector<std::vector<int>> &sel_vertices_counts,
                    std::vector<std::vector<int>> &sel_mask_ids,
                    bool reindex_mask,
                    TensorList &output_list);

   private:
    size_t _user_batch_size;                        ///< Number of samples per batch
    std::vector<std::vector<float>> _output_buffer; ///< Per-sample selected polygon coordinate storage
};
