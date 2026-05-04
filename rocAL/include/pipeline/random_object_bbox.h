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
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "pipeline/content_hash.h"
#include "pipeline/tensor.h"

/*! \brief Per-input cache entry for RandomObjectBbox.
 *
 * Stores the set of foreground labels and the per-class bounding boxes
 * discovered via connected-component labeling. When caching is enabled,
 * repeated inputs with the same content hash reuse these results instead
 * of recomputing them.
 */
struct CacheEntry {
    std::set<int> labels;  ///< Set of foreground label values found in the input
    std::unordered_map<int, std::vector<std::vector<std::vector<unsigned>>>> class_boxes;  ///< Per-label bounding boxes: class_boxes[label][box_idx] = {lo_coords, hi_coords}
    std::unordered_map<int, int> total_boxes;  ///< Per-label total count of connected components

    /*! \brief Retrieve cached bounding boxes for a given label.
     *  \param [out] boxes  Filled with the cached boxes if found.
     *  \param [in]  label  The class label to look up.
     *  \return true if the label was found in the cache, false otherwise.
     */
    bool Get(std::vector<std::vector<std::vector<unsigned>>> &boxes, int label) const {
        auto it = class_boxes.find(label);
        if (it == class_boxes.end())
            return false;
        boxes = it->second;
        return true;
    }

    /*! \brief Store bounding boxes for a given label in the cache.
     *  \param [in] label  The class label.
     *  \param [in] boxes  The bounding boxes to cache.
     */
    void Put(int label, const std::vector<std::vector<std::vector<unsigned>>> &boxes) {
        class_boxes[label] = boxes;
    }
};

/*! \brief Finds bounding boxes of connected components in a segmentation mask and selects one at random.
 *
 * Given a batch of label/segmentation tensors, this class performs connected-component labeling
 * (using a disjoint-set / union-find algorithm) to identify distinct foreground objects, computes
 * their axis-aligned bounding boxes, and randomly selects one per sample. The result is returned
 * as one or two tensors depending on the output format ("anchor_shape", "start_end", or "box").
 *
 * Optionally supports:
 * - Restricting selection to the k-largest objects by volume.
 * - A foreground probability that controls how often a real object is returned vs. the full input extent.
 * - Content-hash-based caching to skip recomputation for repeated inputs.
 */
class RandomObjectBbox {
   public:
    /*! \brief Construct a RandomObjectBbox instance.
     *  \param [in] context          OpenVX context for tensor creation.
     *  \param [in] user_batch_size  Number of samples in each batch.
     *  \param [in] cpu_num_threads  Number of CPU threads available for parallel processing.
     */
    RandomObjectBbox(vx_context context, size_t user_batch_size, size_t cpu_num_threads);
    ~RandomObjectBbox();

    /*! \brief Initialize the operator, allocating output tensors.
     *  \param [in] input          The input label/segmentation tensor.
     *  \param [in] output_format  Output format: "anchor_shape", "start_end", or "box".
     *  \param [in] k_largest      If positive, restrict random selection to the k largest objects by volume.
     *  \param [in] foreground_prob Probability of selecting a foreground object (otherwise returns full image ROI).
     *  \param [in] cache_objects  If true, caches bounding boxes keyed by input content hash.
     *  \return Pointer to the TensorList containing output tensors.
     */
    TensorList *init(Tensor *input, std::string output_format, int k_largest, float foreground_prob, bool cache_objects);

    /*! \brief Re-compute bounding boxes for the current batch of input labels.
     *
     * Called once per iteration before graph execution to populate the output
     * box tensors with a randomly selected object bounding box per sample.
     */
    void update();

    /// Returns a pointer to the raw buffer backing the first output tensor (anchor or start coordinates).
    void *box1_buf() const { return _box1_buf; }
    /// Returns a pointer to the raw buffer backing the second output tensor (shape or end coordinates). May be null for "box" format.
    void *box2_buf() const { return _box2_buf; }

   private:
    /// Scan the input label tensor to collect the set of unique label values present within the ROI.
    template<typename T>
    void findLabels(const T *input, std::set<int> &labels, const std::vector<int> &roi_size, const std::vector<size_t> &max_size);
    /// Produce a binary mask where each element is 1 if the input equals \p label, 0 otherwise.
    template<typename T>
    void filterByLabel(const T *input, std::vector<int> &output, const std::vector<int> &roi_size, const std::vector<size_t> &max_size, int label);
    /// Assign connected-component labels to a single row of the binary mask using run-length encoding.
    void labelRow(const int *label_base, const int *in_row, int *out_row, unsigned length);
    /// Return the group representative for element \p x (identity function — reads the stored group).
    int disjointGetGroup(const int &x) { return x; }
    /// Set element \p x's group to \p new_id and return the previous group.
    int disjointSetGroup(int &x, int new_id);
    /// Find the root representative for element \p x with path compression.
    int disjointFind(int *items, int x);
    /// Merge the sets containing elements \p x and \p y; returns the new root.
    int disjointMerge(int *items, int x, int y);
    /// Merge adjacent rows by unifying component labels at positions where both rows share the same filtered label.
    void mergeRow(int *label_base, const int *in1, const int *in2, int *out1, int *out2, unsigned n);
    /// Core connected-component labeling: filters by a randomly selected label, labels rows, merges across dimensions, and remaps labels to sequential IDs. Returns the total number of connected components.
    template<typename T>
    int labelMergeFunc(const T *input, int &selected_label, std::vector<int> &size, std::vector<size_t> &max_size, std::vector<int> &output_filtered, std::vector<int> &output_compact, std::mt19937 &rng, CacheEntry *cache_entry);
    /// Test and set a bit in the hit bitmap; returns true if the bit was already set.
    bool hit(std::vector<unsigned> &hits, unsigned idx);
    /// Compute or expand axis-aligned bounding boxes from a row of compact labels. Each box spans the min/max coordinates across all dimensions.
    void get_label_boundingboxes(std::vector<std::vector<std::vector<unsigned>>> &boxes, std::vector<std::pair<unsigned, unsigned>> &ranges, std::vector<unsigned> &hits, int *in, const std::vector<int> &origin, unsigned width);
    /// Randomly select a bounding box index, optionally restricted to the k-largest by volume. Returns -1 if no boxes exist.
    int pick_box(const std::vector<std::vector<std::vector<unsigned>>> &boxes, std::mt19937 &rng, int k_largest = -1);

    /*! \brief Per-thread scratch storage for connected-component labeling.
     *
     * Holds the intermediate buffers used by labelMergeFunc during
     * connected-component analysis. One instance is allocated per CPU thread
     * so that OpenMP workers can reuse memory across samples without
     * conflicting with each other. After the first sample processed by a
     * thread, subsequent assign() calls on these vectors skip reallocation
     * when the existing capacity already covers the required size.
     */
    struct ScratchBuffers {
        std::vector<int> output_filtered;  ///< Binary mask produced by filterByLabel (1 where input == selected label)
        std::vector<int> output_compact;   ///< Compact label map produced by labelRow / mergeRow and path-compressed by disjointFind
    };

    vx_context _context;                      ///< OpenVX context used for tensor creation
    size_t _user_batch_size;                   ///< Number of samples per batch
    size_t _cpu_num_threads;                   ///< Number of CPU threads for OMP parallelism
    Tensor *_label_tensor = nullptr;           ///< Non-owning pointer to the input label/segmentation tensor
    Tensor *_box1_tensor = nullptr;            ///< Output tensor for anchor/start/box coordinates (owned by _output_tensor_list)
    Tensor *_box2_tensor = nullptr;            ///< Output tensor for shape/end coordinates (owned by _output_tensor_list, null for "box" format)
    void *_box1_buf = nullptr;                 ///< Raw host buffer backing _box1_tensor, freed in destructor
    void *_box2_buf = nullptr;                 ///< Raw host buffer backing _box2_tensor, freed in destructor
    TensorList _output_tensor_list;            ///< Holds output tensors returned by init()
    std::string _output_format;                ///< Output format: "anchor_shape", "start_end", or "box"
    int _k_largest = -1;                       ///< If positive, restricts selection to the k largest objects
    float _foreground_prob = 1.0f;             ///< Probability of selecting a foreground object
    bool _cache_boxes = false;                 ///< Whether to cache bounding boxes by content hash
    std::unordered_map<content_hash_t, CacheEntry> _boxes_cache;  ///< Content-hash-keyed cache of per-input bounding boxes
    std::vector<ScratchBuffers> _scratch_buffers;  ///< Per-thread scratch storage reused across samples to avoid repeated allocations
};
