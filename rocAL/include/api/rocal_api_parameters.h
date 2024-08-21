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

#ifndef MIVISIONX_ROCAL_API_PARAMETERS_H
#define MIVISIONX_ROCAL_API_PARAMETERS_H
#include "rocal_api_types.h"

/*!
 * \file
 * \brief The AMD rocAL Library - Parameters
 *
 * \defgroup group_rocal_parameters API: AMD rocAL - Parameter API
 * \brief The AMD rocAL Parameters.
 */

/*!
 * \brief  rocalSetSeed
 * \ingroup group_rocal_parameters
 *
 * \param seed
 */
extern "C" void ROCAL_API_CALL rocalSetSeed(unsigned seed);

/*!
 * \brief  rocalGetSeed
 * \ingroup group_rocal_parameters
 *
 * \return
 */
extern "C" unsigned ROCAL_API_CALL rocalGetSeed();

/*!
 * \brief  rocalCreateIntUniformRand
 * \ingroup group_rocal_parameters
 *
 * \param start
 * \param end
 * \return
 */
extern "C" RocalIntParam ROCAL_API_CALL rocalCreateIntUniformRand(int start, int end);

/*!
 * \brief  rocalUpdateIntUniformRand
 * \ingroup group_rocal_parameters
 *
 * \param start
 * \param end
 * \param input_obj
 * \return
 */
extern "C" RocalStatus ROCAL_API_CALL rocalUpdateIntUniformRand(int start, int end, RocalIntParam updating_obj);

/*!
 * \brief  rocalGetIntValue
 * \ingroup group_rocal_parameters
 *
 * \param obj
 * \return
 */
extern "C" int ROCAL_API_CALL rocalGetIntValue(RocalIntParam obj);

/*!
 * \brief  rocalGetFloatValue
 * \ingroup group_rocal_parameters
 *
 * \param obj
 * \return
 */
extern "C" float ROCAL_API_CALL rocalGetFloatValue(RocalFloatParam obj);

/*!
 * \brief  rocalCreateFloatUniformRand
 * \ingroup group_rocal_parameters
 *
 * \param start
 * \param end
 * \return
 */
extern "C" RocalFloatParam ROCAL_API_CALL rocalCreateFloatUniformRand(float start, float end);

/*!
 * \brief  rocalCreateFloatParameter
 * \ingroup group_rocal_parameters
 *
 * \param val
 * \return
 */
extern "C" RocalFloatParam ROCAL_API_CALL rocalCreateFloatParameter(float val);

/*!
 * \brief  rocalCreateIntParameter
 * \ingroup group_rocal_parameters
 *
 * \param val
 * \return
 */
extern "C" RocalIntParam ROCAL_API_CALL rocalCreateIntParameter(int val);

/*!
 * \brief  rocalUpdateFloatParameter
 * \ingroup group_rocal_parameters
 *
 * \param new_val
 * \param input_obj
 * \return
 */
extern "C" RocalStatus ROCAL_API_CALL rocalUpdateFloatParameter(float new_val, RocalFloatParam input_obj);

/*!
 * \brief  rocalUpdateIntParameter
 * \ingroup group_rocal_parameters
 *
 * \param new_val
 * \param input_obj
 * \return
 */
extern "C" RocalStatus ROCAL_API_CALL rocalUpdateIntParameter(int new_val, RocalIntParam input_obj);

/*!
 * \brief  rocalUpdateFloatUniformRand
 * \ingroup group_rocal_parameters
 *
 * \param start
 * \param end
 * \param input_obj
 * \return
 */
extern "C" RocalStatus ROCAL_API_CALL rocalUpdateFloatUniformRand(float start, float end, RocalFloatParam updating_obj);

/*!
 * \brief  rocalCreateIntRand
 * \ingroup group_rocal_parameters
 *
 * \param values
 * \param frequencies
 * \param size
 * \return
 */
extern "C" RocalIntParam ROCAL_API_CALL rocalCreateIntRand(const int *values, const double *frequencies, unsigned size);

/*!
 * \brief  rocalUpdateIntRand
 * \ingroup group_rocal_parameters
 *
 * \param values
 * \param frequencies
 * \param size
 * \param updating_obj
 * \return
 */
extern "C" RocalStatus ROCAL_API_CALL rocalUpdateIntRand(const int *values, const double *frequencies, unsigned size, RocalIntParam updating_obj);

/*!
 * \brief  Sets the parameters for a new or existing RocalFloatRandGen object
 * \ingroup group_rocal_parameters
 * \param values
 * \param frequencies
 * \param size
 * \return
 */
extern "C" RocalFloatParam ROCAL_API_CALL rocalCreateFloatRand(const float *values, const double *frequencies, unsigned size);

/*!
 * \brief  rocalUpdateFloatRand
 * \ingroup group_rocal_parameters
 *
 * \param values
 * \param frequencies
 * \param size
 * \param updating_obj
 * \return
 */
extern "C" RocalStatus ROCAL_API_CALL rocalUpdateFloatRand(const float *values, const double *frequencies, unsigned size, RocalFloatParam updating_obj);

#endif // MIVISIONX_ROCAL_API_PARAMETERS_H
