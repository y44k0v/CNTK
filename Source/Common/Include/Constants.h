// Constants.h -- the constants used by CNTK
//

#pragma once

#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

// Constants used in SGD distributed gradient aggregation.
// The default threshold size to pack a gradient into a continuous buffer during aggregation for less MPI ops.
const size_t DEFAULT_PACK_THRESHOLD_SIZE_IN_KB = 32;
const size_t DEFAULT_PACK_THRESHOLD_SIZE_IN_BYTES = DEFAULT_PACK_THRESHOLD_SIZE_IN_KB * 1024;

// version number to control how to read and write
#define CNTK_MODEL_VERSION_1 1
#define CNTK_MODEL_VERSION_2 2
#define CNTK_MODEL_VERSION_3 3
#define CNTK_MODEL_VERSION_4 4   // PastValue
#define CNTK_MODEL_VERSION_5 5   // ND convolution and pooling
#define CNTK_MODEL_VERSION_6 6   // batch-norm blending
#define CNTK_MODEL_VERSION_7 7   // ElemType tag in model file
#define CNTK_MODEL_VERSION_8 8   // DynamicAxis for inputs
#define CNTK_MODEL_VERSION_9 9   // transpose flag in ConvolutionNode to support deconvolution
#define CNTK_MODEL_VERSION_10 10 // learning-rate multiplier for input nodes
#define CNTK_MODEL_VERSION_11 11 // dynamic axis name for where nodes
#define CNTK_MODEL_VERSION_12 12 // Times() m_inputRank to support parameter-rank inference
#define CNTK_MODEL_VERSION_13 13 // batch norm: switch running inverse std deviation -> variance, MB count -> samplesSeen; CuDNN v5
#define CNTK_MODEL_VERSION_14 14 // axis parameter in OptimizedRNNStackNode
#define CNTK_MODEL_VERSION_15 15 // add new nodes: LambdaRankNode and NDCG1Eval
#define CNTK_MODEL_VERSION_16 16 // save/load rng state for Dropout and RandomSample nodes.
#define CNTK_MODEL_VERSION_17 17 // use 8 bytes for rng seeds on both platforms
#define CNTK_MODEL_VERSION_18 18 // reserving 18 for dilated convolution, write out one more TensorShape 
#define CNTK_MODEL_VERSION_19 19 // batch norm: flag whether running mean count is 0
#define CNTK_MODEL_VERSION_20 20 // adding output shape to convolution node
#define CNTK_MODEL_VERSION_21 21 // pooling: add a ceilOutDim to decide whether ceil or floor while computing the output size
#define CNTK_MODEL_VERSION_22 22 // Slice and pad accepts multiple axes 
#define CNTK_MODEL_VERSION_23 23 // pooling: add include pad func for average pooling
#define CNTK_MODEL_VERSION_24 24 // ReduceElements: add keepDimensions
#define CNTK_MODEL_VERSION_25 25 // transpose: allow specifying a permutation
#define CURRENT_CNTK_MODEL_VERSION CNTK_MODEL_VERSION_25

#endif
