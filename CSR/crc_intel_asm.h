// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

// Implementations adapted from Intel's Slicing By 8 Sourceforge Project
// http://sourceforge.net/projects/slicing-by-8/
/*
 * Copyright 2016 Ferry Toth, Exalon Delft BV, The Netherlands
 *
 *
 * This software program is licensed subject to the BSD License,
 * available at http://www.opensource.org/licenses/bsd-license.html.
 *
 * Abstract:
 *
 *  This file is just a C wrapper around Intels assembly optimized crc_pcl
 */
#include <stdint.h>

uint32_t crc_pcl(const uint8_t * buffer, uint32_t len, uint32_t crc_init);
